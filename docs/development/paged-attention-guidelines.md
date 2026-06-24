# Linee guida per evolvere la Paged Attention verso un comportamento stile vLLM

Questo documento guida i modelli/agent che continueranno a modificare il progetto per rendere il runtime più vicino al modello vLLM/PagedAttention. È complementare a `docs/paged-scheduler-migration-plan.md`: quel documento copre soprattutto il server scheduler; questo copre il core KV cache, il grafo di attention e le regole operative da rispettare.

## Stato attuale osservato dal grafo

Il progetto contiene già un supporto sperimentale paged KV, non solo uno stub:

- `src/llama-kv-cache-paged.h` definisce `llama_kv_block`, `llama_kv_block_allocator` e `llama_kv_block_table`.
- `include/llama.h` espone `llama_context_params::paged_kv` e `kv_block_size` come opzioni sperimentali.
- `src/llama-context.cpp` propaga `params.paged_kv` e normalizza `kv_block_size`.
- `src/llama-kv-cache.h` mantiene allocator per stream, block table, refcount/COW stats e input tensor paged.
- `src/llama-kv-cache.cpp` contiene già un fast path in `find_slot()` che, quando `paged` è attivo, risolve token → cella tramite `(seq_id, logical_page) → block_id` e pianifica nuovi blocchi/COW.
- `src/llama-kv-cache.cpp::apply_ubatch()` applica i piani COW e aggiorna i metadati cella/block table.
- `src/llama-graph.cpp::build_attn_mha()` passa `block_table`, `seq_ids_q`, `page_limits_q` a `ggml_flash_attn_ext` quando Flash Attention è in uso.
- I path ISWA/hybrid propagano i tensori paged per base/SWA dove presenti.

Quindi i prossimi passi non devono ricominciare da zero: devono consolidare e completare l’integrazione tra block allocator, attention backend, server scheduler e API pubbliche.

## Obiettivo architetturale

Rendere la modalità paged una modalità runtime coerente, dove:

1. Le sequenze non richiedono slot fisici contigui equivalenti al massimo contesto.
2. La KV cache è allocata a blocchi/pagine e indirizzata tramite block table.
3. Prefill, decode, fork/copy di sequenze e prefix reuse funzionano tramite ownership/refcount dei blocchi.
4. L’admission control del server usa capacità reale in blocchi liberi/riservati, non il vecchio modello a slot.
5. L’attention backend legge K/V tramite block table senza dipendere dal layout contiguo legacy.
6. Il path legacy resta compatibile, ma non deve contaminare la semantica paged.

## Invarianti da non rompere

- `kv_block_size` deve restare una potenza di due positiva.
- `block_id` identifica un blocco fisico dentro la slab piatta esistente; non sostituire la slab senza una migrazione esplicita.
- `logical_page = pos / block_size`; `intra_offset = pos % block_size`.
- La block table è la fonte della mappatura logica in paged mode: `(seq_id, page) -> block_id`.
- Una pagina condivisa da `seq_cp()` deve essere protetta da refcount e mutata solo dopo COW.
- `seq_id` è l’identità runtime llama; non confonderlo con `id_slot` o lineage/session key del server.
- In paged mode, celle non logicamente presenti devono restare mascherate; bytes stale nei buffer fisici non devono diventare osservabili.
- Ogni modifica a `find_slot()`, `apply_ubatch()`, `seq_rm()`, `seq_cp()`, `seq_keep()`, state load/save o kq mask deve mantenere block table e cell metadata consistenti.
- Ogni modifica a ISWA/hybrid deve considerare sia cache base sia cache SWA.
- Il fallback non paged deve continuare a funzionare e non deve pagare costi del path paged quando `paged_kv == false`.

## Roadmap consigliata

### 1. Consolidare il core allocator/block table

Priorità:

- Aggiungere test mirati per `llama_kv_block_allocator` e `llama_kv_block_table`.
- Coprire almeno: alloc/free, refcount, retain/release, `erase_seq`, `erase_page`, sparse pages, partial last block, `max_mapped_page_plus1`.
- Verificare che `seq_rm()` liberi blocchi solo quando tutte le celle del blocco sono vuote.
- Verificare COW su `seq_cp()` seguito da scrittura su una pagina condivisa.
- Verificare `rebuild_block_table_for_seq()` dopo restore stato.

Regola: prima di cambiare policy di scheduling, rendere deterministica la struttura dati.

### 2. Rendere `find_slot()` block-first in paged mode

Il codice contiene già un fast path block-table-driven. I prossimi modelli devono:

- Evitare di reintrodurre scansioni ring-buffer come comportamento principale in `paged` mode.
- Trattare il ring-buffer come fallback temporaneo o path legacy.
- Assicurare che le nuove pagine siano pianificate con `peek_free()` e poi acquisite una sola volta in `apply_ubatch()`/record path.
- Impedire collisioni cella tra sequenze diverse, soprattutto con `kv_unified`.
- Mantenere COW atomico rispetto al piano di ubatch: se manca capacità per una pagina nuova o COW, fallire la preparazione senza mutare stato parziale.

Criterio di completamento: in un decode paged normale, l’allocazione dei token nuovi deve derivare da page/block planning, non da ricerca casuale di celle libere.

### 3. Completare input tensors per attention paged

Verificare e stabilizzare:

- `build_input_block_table()` / `set_input_block_table()`.
- `build_input_seq_ids_q()` / `set_input_seq_ids_q()`.
- `build_input_page_limits_q()` / `set_input_page_limits_q()`.
- Reuse del grafo quando aumenta `max_mapped_page_plus1`.
- Shape dei tensori ISWA base/SWA.

Regola: se una modifica cambia shape o semantica di questi tensori, aggiornare sia `llm_graph_input_attn_kv` sia `llm_graph_input_attn_kv_iswa` e i path hybrid.

### 4. Portare i backend Flash Attention a leggere davvero a pagine

Il grafo passa già metadata paged a `ggml_flash_attn_ext`. I prossimi passi sono backend-specific:

- Verificare quali backend usano `block_table`, `seq_ids_q`, `page_limits_q` realmente e quali li ignorano.
- Implementare/validare gather K/V per pagine nel backend target, iniziando da Metal se è quello già previsto dai commenti.
- Mantenere fallback corretto se il backend non supporta paged FA: errore chiaro o path legacy verificato, non risultati silenziosamente sbagliati.
- Aggiungere test o golden checks su output equivalenti tra non-paged e paged con stesso prompt/batch.

Criterio di completamento: `paged_kv + flash_attn` deve produrre output equivalenti al path non paged per casi piccoli e deve non leggere fuori dai blocchi mappati.

### 5. Allineare server scheduler al modello a blocchi

Seguire `docs/paged-scheduler-migration-plan.md`. In più:

- L’admission deve ragionare in blocchi necessari per prefill/decode, blocchi riservati e blocchi liberi reali (`llama_kv_cache_n_free_blocks`).
- Prefix cache e lineage devono operare su sequenze/blocchi, non su slot fisici.
- Save/restore/erase in paged mode devono chiamare API coerenti con block table e rebuild quando serve.
- Non usare `server_slot` come stato runtime paged; se serve compatibilità API, usare adapter espliciti.

### 6. Osservabilità e debug

Aggiungere metriche utili per capire frammentazione e comportamento paged:

- blocchi totali/liberi/usati/riservati;
- blocchi condivisi/refcount > 1;
- COW count e bytes copiati (`paged_cow_stats`);
- pagine mappate per seq;
- fallimenti admission per capacità insufficiente;
- fallback da paged fast path a ring-buffer.

Le metriche devono distinguere chiaramente:

- capacità fisica della KV cache;
- capacità riservata dal server scheduler;
- capacità effettivamente mappata nella block table.

## Regole operative per i modelli/agent futuri

### Prima di modificare

1. Usare il grafo per localizzare simboli e chiamanti.
2. Leggere il codice reale prima di proporre o applicare cambiamenti.
3. Identificare se la modifica tocca:
   - core KV (`src/llama-kv-cache.*`, `src/llama-kv-cache-paged.h`),
   - graph/attention (`src/llama-graph.*`),
   - backend ggml,
   - server scheduler (`tools/server/*`),
   - API pubbliche (`include/llama.h`).
4. Controllare callers/callees prima di cambiare funzioni condivise come `find_slot()`, `apply_ubatch()`, `set_input_kq_mask()`, `build_attn_mha()`.

### Durante le modifiche

- Fare cambi piccoli, verificabili e monotoni.
- Non aggiungere astrazioni generiche se una modifica locale basta.
- Non cambiare contemporaneamente scheduler server e attention backend nello stesso commit, salvo necessità stretta.
- Non introdurre compatibilità fittizia: se una feature non è supportata in paged mode, restituire errore esplicito.
- Non mascherare failure di capacity con fallback silenziosi a comportamento slot-based.
- Non assumere che `n_stream == n_seq_max`: in unified KV `n_stream == 1`.
- Non assumere che tutte le architetture usino solo KV attention standard: considerare ISWA, hybrid recurrent, MLA e no-cache embedding/diffusion.

### Dopo le modifiche

Eseguire almeno una verifica mirata:

- test unitario se si tocca allocator/block table;
- test small decode se si tocca `find_slot()`/`apply_ubatch()`;
- confronto output paged vs non-paged se si tocca attention;
- test server/admission se si tocca scheduler.

Se non esiste un test, aggiungere almeno un test piccolo o documentare esplicitamente cosa non è verificabile.

## Feature gap da chiudere

- Copertura test insufficiente per allocator, block table e COW.
- Verifica backend-specific dell’uso reale dei metadata paged in `ggml_flash_attn_ext`.
- Semantica save/restore paged da rendere esplicita e testata con rebuild block table.
- Server paged ancora da separare completamente dal modello slot legacy.
- Policy di eviction/prefix cache da allineare al refcount dei blocchi.
- Metriche paged da rendere abbastanza complete per diagnosticare frammentazione e capacity.

## Criteri di accettazione finali

Una fase può considerarsi “vLLM-like” solo quando:

1. Più richieste concorrenti condividono il pool KV a blocchi senza slot dedicati full-context.
2. Il prefix reuse condivide blocchi tramite refcount e usa COW alla prima scrittura.
3. L’admission accetta/rifiuta richieste in base a blocchi disponibili e riservati.
4. L’attention legge K/V tramite block table nei backend supportati.
5. La modalità paged ha test di equivalenza su casi piccoli contro il path non paged.
6. I fallback sono espliciti, misurabili e non nascondono risultati non corretti.
