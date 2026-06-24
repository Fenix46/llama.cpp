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

Il grafo passa già metadata paged a `ggml_flash_attn_ext` (`block_table` = src[5], `seq_ids_q` = src[6], `page_limits_q` = src[7]). Stato verificato per backend (vedi sotto). I prossimi passi sono backend-specific:

- Implementare/validare gather K/V per pagine nei backend non ancora coperti.
- Mantenere fallback corretto se il backend non supporta paged FA: errore chiaro o path legacy verificato, non risultati silenziosamente sbagliati.
- Aggiungere test o golden checks su output equivalenti tra non-paged e paged con stesso prompt/batch.

Criterio di completamento: `paged_kv + flash_attn` deve produrre output equivalenti al path non paged per casi piccoli e deve non leggere fuori dai blocchi mappati.

#### Mappa del gating paged FA (verificata)

Switch globale: env `LLAMA_PAGED_ATTN`. Default attivo; `LLAMA_PAGED_ATTN=0` forza il path legacy. Questo è anche lo strumento diagnostico per isolare il rumore numerico del kernel paged (vedi sotto e `tests/test-paged-kv-equiv.cpp`).

**Metal** (`ggml/src/ggml-metal/ggml-metal-ops.cpp`, ~riga 2705):
- Usa il kernel paged (`kernel_flash_attn_ext_paged` o `_paged_vec`) solo se TUTTE: `src[5]!=nullptr`, `LLAMA_PAGED_ATTN!=0`, `src[1]->type == F16`, K/V contigui in F16 (`nb10/nb20 == sizeof(F16)`), V head dim `ne20 <= 576` (= 18*32).
- Variante `_paged_vec` solo per: maschera presente, no sinks/bias/softcap, `ne00==64 && ne20==64`.
- Se i vincoli paged non sono soddisfatti: **fallback silenzioso al kernel legacy** sulla stessa slab piatta. È corretto perché in paged mode la slab fisica resta contigua F16, quindi il kernel legacy legge gli stessi byte — ma NON applica il mascheramento per-pagina del block table. Affidarsi al fatto che le celle non mappate restino mascherate dalla kq mask.
- Quando paged attivo, asserisce che src[6] e src[7] siano presenti (hard error, non garbage).

**CUDA** (`ggml/src/ggml-cuda/fattn.cu`, ~riga 334):
- `paged_attn_active = block_table != nullptr && LLAMA_PAGED_ATTN!=0`.
- Se `block_table != nullptr` ma `LLAMA_PAGED_ATTN=0` → `BEST_FATTN_KERNEL_NONE` (**errore esplicito**: forzare il legacy su layout paged darebbe garbage). Più rigoroso di Metal.
- Se paged attivo ma manca src[6] o src[7] → `BEST_FATTN_KERNEL_NONE`.
- Scelta kernel via env `LLAMA_PAGED_KERNEL` = `tile` | `mma` | `auto` (default auto). `auto`/`mma` usano MMA_F16 solo se shape supportata (`paged_mma_shape_supported`: Q head dim in {64,80,96,112,128,256,320,512,576} con vincoli V/GQA) e hardware Turing/Volta/AMD-MFMA; altrimenti fallback a TILE.

**Nota numerica (verificata, gemma-4-E2B IQ3_XXS su Metal):** col kernel paged FA attivo i logit divergono da quelli non-paged di ~1e-2 (max abs). Con `LLAMA_PAGED_ATTN=0` la differenza scende a 0. È rumore di accumulo del kernel, non un bug del data path: con FA disabilitato paged e non-paged sono bit-identici. I test di equivalenza devono usare tolleranze separate (stretta per il data path, larga per il kernel paged FA).

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
