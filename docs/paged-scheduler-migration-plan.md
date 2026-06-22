# Migrazione completa da slot scheduler a paged scheduler

## Contesto

Il server oggi contiene due modelli runtime paralleli:

- `server_slot`, runtime legacy slot-based, definito in `tools/server/server-context.cpp`.
- `paged_request_state`, runtime paged, definito in `tools/server/paged-request.h`.

Il path paged non sostituisce ancora completamente gli slot. È innestato dentro `server_context_impl` con branch dedicati, mentre varie API e operazioni interne restano slot-centriche.

Punti osservati nel codice:

- `server_context_impl` mantiene sia `slots` sia `paged_requests`.
- Anche in paged mode gli slot vengono inizializzati.
- `update_slots()` contiene un branch paged con early return, seguito dal vecchio path slot-based.
- `id_slot` viene ancora usato come concetto API, e in paged mode viene riutilizzato semanticamente come lineage key.
- Le azioni `slot_save`, `slot_restore`, `slot_erase` operano ancora su `server_slot`.
- `kv_block_scheduler` è un componente di metriche/osservabilità, non uno scheduler di esecuzione.

## Obiettivo

Arrivare a un’architettura in cui:

1. Il path paged non usa `server_slot` per eseguire richieste.
2. La selezione dello scheduler avviene tramite un backend esplicito, non tramite branch profondi in `server-context.cpp`.
3. Le API pubbliche legacy restano compatibili dove necessario, ma internamente non impongono più il modello a slot.
4. Lifecycle, lease seq-id, prefix cache, block manager e metrics hanno ownership chiara.
5. Il feature gap del paged scheduler è esplicito: ogni feature è supportata, non supportata con errore chiaro, oppure pianificata.

## Non-obiettivi iniziali

- Non fare un big-bang rewrite di `server-context.cpp`.
- Non rimuovere subito il runtime legacy `server_slot`.
- Non cambiare API pubbliche senza una fase di compatibilità.
- Non forzare una classe comune unica se aumenta la complessità.

## Invarianti target

### Invarianti paged mode

- `paged_request_state` è l’owner dello stato runtime per richieste paged.
- `server_slot` non deve essere consultato per launch, decode, cancel, save/restore/erase o metrics operative di richieste paged.
- `seq_id` identifica la sequenza llama runtime.
- `lineage_key` identifica continuità logica/sessione, non deve essere confuso con `id_slot`.
- Il lease di `seq_id` è gestito da un solo componente responsabile.
- Il lifecycle di una request paged passa da un solo componente responsabile delle transizioni.

### Invarianti legacy slot mode

- Il backend legacy può continuare a usare `server_slot`.
- Compatibilità con endpoint e metriche slot-based va mantenuta finché il backend legacy esiste.
- Le modifiche al paged backend non devono cambiare il comportamento legacy se non esplicitamente pianificato.

## Architettura target

Introdurre una boundary interna tra server/API layer e scheduler runtime.

Esempio di interfaccia concettuale:

```cpp
class ServerSchedulerBackend {
public:
    virtual ~ServerSchedulerBackend() = default;

    virtual bool launch(server_task && task) = 0;
    virtual void cancel(int task_id) = 0;
    virtual void tick() = 0;

    virtual int32_t active_count() const = 0;
    virtual int32_t idle_count() const = 0;

    virtual json metrics(bool debug) const = 0;

    virtual bool save_state(int handle_id, const std::string & path, server_task_result_slot_save_load & out) = 0;
    virtual bool restore_state(int handle_id, const std::string & path, server_task_result_slot_save_load & out) = 0;
    virtual bool erase_state(int handle_id, server_task_result_slot_erase & out) = 0;
};
```

Il nome e la forma finale possono cambiare. Il punto importante è che `server_context_impl` non deve più contenere direttamente tutta la logica di due scheduler diversi.

Backend iniziali:

- `SlotSchedulerBackend`: wrapper sul codice legacy esistente.
- `PagedSchedulerBackend`: wrapper sul codice paged esistente.

## Fasi operative

---

## Fase 0 — Baseline e invarianti

### Scopo

Stabilire il comportamento attuale e le regole target prima di muovere codice.

### Task

- [ ] Mappare le funzioni legacy slot-critical:
  - launch
  - cancel
  - update/decode
  - save/restore/erase
  - metrics
  - prompt cache
  - prefix cache
- [ ] Mappare le funzioni paged-critical:
  - request allocation
  - seq lease
  - admission
  - launch
  - lifecycle
  - prefill/decode tick
  - release/cache
  - metrics
- [ ] Elencare API pubbliche che espongono ancora concetti slot.
- [ ] Stabilire quali API devono restare compatibili.
- [ ] Stabilire quali API possono restituire errore chiaro in paged mode.

### Criteri di completamento

- [ ] Documento aggiornato con mappa dei punti di ingresso.
- [ ] Invarianti paged confermati.
- [ ] Feature gap iniziale elencato.

---

## Fase 0.1 — Mappa estrazione `server-context.cpp`

### Scopo

Trasformare la diagnosi in una mappa operativa dei simboli reali da estrarre. Questa fase evita di progettare un’interfaccia backend astratta senza conoscere le dipendenze concrete.

### Entry point principali

| Area | Simbolo | Posizione | Destinazione target |
| --- | --- | --- | --- |
| Dispatch task | `process_single_task()` | `tools/server/server-context.cpp:3408` | resta orchestratore temporaneo, poi dispatch a backend |
| Tick runtime | `update_slots()` | `tools/server/server-context.cpp:3819` | diventa `backend->tick()` |
| Legacy launch | `launch_slot_with_task()` | `tools/server/server-context.cpp:2213` | `SlotSchedulerBackend` |
| Legacy parent/child launch | `launch_slots_with_parent_task()` | `tools/server/server-context.cpp:3325` | `SlotSchedulerBackend` |
| Legacy slot selection | `get_available_slot()` | `tools/server/server-context.cpp:1747` | `SlotSchedulerBackend` |
| Legacy slot lookup | `get_slot_by_id()` | `tools/server/server-context.cpp:1521` | legacy backend / compatibility adapter |
| Paged allocation | `get_or_create_paged_request()` | `tools/server/server-context.cpp:1655` | `PagedSchedulerBackend` o `PagedRequestAllocator` |
| Paged lineage reuse | `try_lineage_reuse()` | `tools/server/server-context.cpp:1586` | `PagedSchedulerBackend` / lineage adapter |
| Paged launch | `launch_paged_request()` | `tools/server/server-context.cpp:2335` | `PagedSchedulerBackend` |
| Paged admission | `paged_admission_decision()` | `tools/server/server-context.cpp:3285` | `PagedSchedulerBackend` / admission layer |
| Paged admission bool | `paged_admission_available()` | `tools/server/server-context.cpp:3320` | wrapper backend |
| Paged request lookup | `get_paged_request_by_seq_id()` | `tools/server/server-context.cpp:1971` | `PagedSchedulerBackend` |
| Paged idle eviction | `try_clear_idle_paged_requests()` | `tools/server/server-context.cpp:1938` | `PagedSchedulerBackend` / block cache manager |
| Prefix reuse launch | `execute_prefix_reuse_plan()` | `tools/server/server-context.cpp:2142` | `PagedSchedulerBackend` / prefix reuse adapter |
| Prefix release cache | `register_paged_prefix_cache_on_release()` | `tools/server/server-context.cpp:2039` | `RequestLifecycle` / paged backend |
| Shared LoRA helper | `construct_lora_list()` | `tools/server/server-context.cpp:2200` | shared helper or backend dependency |
| Slot actions | `SERVER_TASK_TYPE_SLOT_SAVE/RESTORE/ERASE` branch | `tools/server/server-context.cpp:3659` | backend state API / compatibility adapter |

### Paged path dependency map

`launch_paged_request()` currently calls:

- `construct_lora_list()`
- `prepare_empty_sequence_for_prefix_copy()`
- `send_error()`
- `paged_task_reserved_blocks()`
- `register_paged_prefix_cache_on_release()`
- `get_paged_request_by_seq_id()`
- `reset_runtime_state_for_new_request()`
- `execute_prefix_reuse_plan()`

Implication: il primo backend paged non può essere completamente puro. Deve ricevere callback o references per:

- error reporting (`send_error`)
- response sending/finalization
- LoRA construction/config
- prefix cache/lifecycle operations
- metrics
- queue/task defer
- model/context pointers

`get_or_create_paged_request()` currently calls:

- `paged_task_reserved_blocks()`
- `prefix_cache_invalidate()`
- `try_clear_idle_paged_requests()`

Dipendenze da incapsulare:

- `paged_requests`
- `paged_seq_leases`
- `params_base`
- `ctx`
- `n_ctx_slot_`
- `paged_blocks_per_seq_`
- `paged_max_full_ctx_concurrency_`
- `BlockManager`
- `LineageManager`
- `PrefixReuseManager`

`paged_admission_decision()` currently calls:

- `BlockManager::stats()`
- `BlockManager::total_reserved_blocks()`
- `llama_kv_cache_n_free_blocks()`
- `try_clear_idle_paged_requests()`
- `paged_admission_with_capacity()`

Dipendenze da incapsulare:

- capacity state
- KV free block query
- idle eviction callback
- params/admission policy

`update_slots()` paged branch currently calls, among others:

- `paged_admission_decision()`
- `get_paged_request_by_seq_id()`
- `mark_request_uncacheable()`
- `reset_paged_request_for_reprefill()`
- `send_final_response()`
- `send_error()`
- `create_checkpoint()`
- `send_partial_response()`
- `send_embedding()`
- `send_rerank()`
- `populate_token_probs()`
- `process_token()`
- `count_paged_reserved_blocks()`
- `try_clear_idle_paged_requests()`

Implication: l’estrazione del tick deve essere fatta dopo allocation/launch, oppure con un backend che espone callback verso `server_context_impl` per response e sampling side effects.

### Legacy path dependency map

`process_single_task()` legacy calls:

- `get_slot_by_id()`
- `get_available_slot()`
- `get_free_slots()`
- `launch_slots_with_parent_task()`
- `launch_slot_with_task()`
- `slot_save_and_clear()`

`launch_slot_with_task()` currently calls:

- `construct_lora_list()`
- `send_error()`

Legacy extraction is simpler than paged extraction because state ownership is mostly inside `server_slot`, but slot actions and prompt cache remain coupled to `server_context_impl`.

### Classificazione iniziale

#### Resta temporaneamente in `server_context_impl`

- HTTP/API response construction.
- `send_error()`.
- `send_partial_response()`.
- `send_final_response()`.
- `send_embedding()`.
- `send_rerank()`.
- `populate_token_probs()` until sampling/output ownership is separated.
- Queue wiring in `init()`.
- Model loading and global params initialization.

#### Primo blocco da estrarre nel paged backend

Stato: completato con `server_scheduler::PagedRequestAllocator` in `tools/server/scheduler/paged_request_allocator.{h,cpp}`. `server_context_impl` mantiene wrapper sottili per limitare il diff e preservare i call site esistenti.

- [x] `paged_requests` ownership wrapper.
- [x] `paged_seq_leases` ownership wrapper.
- [x] `get_paged_request_by_seq_id()`.
- [x] `try_clear_idle_paged_requests()`.
- [x] `get_or_create_paged_request()`.
- [x] `paged_admission_decision()`.
- [x] `paged_admission_available()`.

Reason: questi componenti formano il nucleo allocation/admission e hanno meno dipendenze da response streaming rispetto al decode tick.

#### Secondo blocco da estrarre nel paged backend

Stato parziale: prefix reuse estratto in `server_scheduler::PagedPrefixReuse` in `tools/server/scheduler/paged_prefix_reuse.{h,cpp}`. `server_context_impl` mantiene wrapper sottili per `build_prefix_reuse_metadata()` ed `execute_prefix_reuse_plan()`.

- [ ] `launch_paged_request()`.
- [x] `execute_prefix_reuse_plan()`.
- [x] `build_prefix_reuse_metadata()`.
- [ ] `register_paged_prefix_cache_on_release()` solo se prima si chiarisce ownership con `RequestLifecycle`.

Reason: launch introduce dipendenze su LoRA, sampler, prefix cache e error reporting; va fatto dopo il nucleo allocation.

#### Terzo blocco da estrarre nel paged backend

- Branch paged di `update_slots()`.
- Policy prefill/decode.
- Decode callbacks.
- Sampling callbacks.
- Metrics paged.

Reason: è il blocco più accoppiato a response queue, token processing e metriche.

#### Blocchi legacy da estrarre dopo boundary iniziale

- `get_available_slot()`.
- `get_free_slots()`.
- `launch_slot_with_task()`.
- `launch_slots_with_parent_task()`.
- branch legacy di `update_slots()`.

Reason: il path legacy deve restare stabile mentre si isola il paged path.

### Prima interfaccia minima consigliata

Per evitare un’interfaccia troppo grande all’inizio, introdurre prima un backend minimo:

```cpp
class ServerSchedulerBackend {
public:
    virtual ~ServerSchedulerBackend() = default;
    virtual bool launch(server_task && task) = 0;
    virtual void cancel(int task_id) = 0;
    virtual void tick() = 0;
    virtual int32_t active_count() const = 0;
};
```

Le operazioni `metrics`, `save_state`, `restore_state`, `erase_state` vanno aggiunte dopo la mappatura `/slots`, perché oggi hanno semantica ancora ambigua in paged mode.

### Task Fase 0.1

- [x] Confermare questa mappa contro il codice prima della Fase 1.
- [x] Decidere il nome del primo componente e il path dei nuovi file: `server_scheduler::PagedRequestAllocator` in `tools/server/scheduler/`.
- [x] Decidere se `paged_requests` rimane inizialmente in `server_context_impl` passato per reference, oppure viene subito spostato nel backend paged: rimane in `server_context_impl` ed è passato per reference all’allocator.
- [ ] Decidere la strategia iniziale per callback response/error.

### Criteri di completamento

- [ ] La prima PR/refactor può limitarsi ad aggiungere backend minimo senza cambiare comportamento.
- [ ] Allocation/admission paged hanno una destinazione chiara.
- [ ] Tick/decode paged è riconosciuto come fase successiva, non come primo taglio.

---

## Fase 1 — Introdurre backend scheduler comune

### Scopo

Creare una boundary senza cambiare comportamento.

### Task

- [ ] Definire interfaccia interna `ServerSchedulerBackend` o nome equivalente.
- [ ] Definire tipo handle interno separato da `id_slot`.
- [ ] Creare `SlotSchedulerBackend` come thin wrapper del path legacy.
- [ ] Creare `PagedSchedulerBackend` come thin wrapper del path paged.
- [ ] Spostare la selezione `scheduler == "paged"` verso init/configurazione backend.
- [ ] Lasciare temporaneamente implementazioni delegate a `server_context_impl` se necessario.

### Criteri di completamento

- [ ] Il server compila.
- [ ] Legacy scheduler mantiene completions base.
- [ ] Paged scheduler mantiene completions base.
- [ ] `server_context_impl` sceglie un backend, anche se la logica è ancora parzialmente delegata.

---

## Fase 2 — Estrarre allocation/admission paged

### Scopo

Togliere da `server_context_impl` la responsabilità diretta di creare e ammettere request paged.

### Task

- [ ] Spostare `get_or_create_paged_request()` nel backend paged o in un componente dedicato.
- [ ] Spostare `try_lineage_reuse()` fuori da `server_context_impl`.
- [ ] Spostare `paged_admission_decision()` e `paged_admission_available()` nel backend paged/admission layer.
- [ ] Rendere esplicite le dipendenze richieste:
  - `ctx`
  - `params_base`
  - `paged_requests`
  - `paged_seq_leases`
  - `BlockManager`
  - `LineageManager`
  - `PrefixReuseManager`
- [ ] Aggiungere test/manual repro per capacity reached e KV exhausted.

### Criteri di completamento

- [ ] `process_single_task()` non contiene più dettagli di lease/admission paged.
- [ ] Defer per paged capacity funziona come prima.
- [ ] Reuse/cached seq non cambia comportamento.

---

## Fase 3 — Estrarre launch paged

### Scopo

Rendere `launch_paged_request()` responsabilità del backend paged.

### Task

- [ ] Spostare logica LoRA/aLoRA paged in helper condiviso o backend paged.
- [ ] Spostare validazione token paged.
- [ ] Spostare setup sampler paged.
- [ ] Spostare setup callback release paged.
- [ ] Spostare `execute_prefix_reuse_plan()` o incapsularlo nel backend paged.
- [ ] Separare error reporting server da logica backend con callback dedicate.

### Criteri di completamento

- [ ] Launch paged non dipende direttamente da `server_slot`.
- [ ] Launch legacy e launch paged sono chiaramente separati.
- [ ] Errori sampler/token validation sono propagati correttamente.

---

## Fase 4 — Estrarre tick/decode paged da `update_slots()`

### Scopo

Rimuovere il grosso branch paged da `update_slots()`.

### Task

- [ ] Creare metodo `PagedSchedulerBackend::tick()`.
- [ ] Spostare sweep TTL/LRU paged.
- [ ] Spostare costruzione schedule decision.
- [ ] Spostare policy prefill/decode.
- [ ] Spostare callbacks di prefill.
- [ ] Spostare callbacks di decode/sampling.
- [ ] Spostare gestione `kv_sched` metrics per paged.
- [ ] Spostare gestione empty-turn/stall.
- [ ] Lasciare `server_context_impl::update_slots()` come dispatch a `backend->tick()`.

### Criteri di completamento

- [ ] `update_slots()` non contiene più logica paged interna.
- [ ] Paged completions base funzionano.
- [ ] Streaming paged funziona.
- [ ] Cancel durante decode funziona.
- [ ] Metrics decode/prefill ancora aggiornate.

---

## Fase 5 — Normalizzare handle, `id_slot`, `seq_id`, `lineage_key`

### Scopo

Eliminare ambiguità semantiche interne.

### Task

- [ ] Introdurre tipo interno per handle request/session.
- [ ] Separare chiaramente:
  - API `id_slot` legacy
  - runtime `seq_id`
  - logical `lineage_key`
  - task id
- [ ] Rimuovere uso di `id_slot` come lineage key nella logica paged interna.
- [ ] Spostare mapping API compatibility in un adapter HTTP/server layer.
- [ ] Aggiornare logging per distinguere `slot_id`, `seq_id`, `request_id`, `lineage_key`.

### Criteri di completamento

- [ ] Nessuna funzione paged interna interpreta `id_slot` come lineage key.
- [ ] Log paged leggibili e non ambigui.
- [ ] API legacy continua a funzionare o restituisce errore documentato.

---

## Fase 6 — Migrare `/slots` e slot actions

### Scopo

Rendere save/restore/erase indipendenti dal runtime legacy.

### Decisione richiesta

Per paged mode scegliere una delle opzioni:

1. Supporto reale su `paged_request_state`/`seq_id`.
2. Endpoint compatibile che mappa request/seq handle.
3. Errore esplicito: unsupported in paged scheduler.

### Task

- [ ] Definire comportamento ufficiale per paged mode.
- [ ] Se supportato:
  - [ ] Implementare save state da `paged_request_state`.
  - [ ] Implementare restore state su `paged_request_state` con lease corretto.
  - [ ] Implementare erase state su `paged_request_state` e block manager.
- [ ] Se non supportato:
  - [ ] Restituire errore chiaro prima di chiamare `get_slot_by_id()`.
- [ ] Aggiornare metrics slots/request data.
- [ ] Aggiornare Web UI/API types se necessario.

### Criteri di completamento

- [ ] Nessuna slot action paged passa da `server_slot`.
- [ ] Comportamento documentato.
- [ ] Errori chiari per feature non supportate.

---

## Fase 7 — Unificare lifecycle e ownership paged

### Scopo

Ridurre callback sparse e rendere `RequestLifecycle` il punto centrale delle transizioni paged.

### Task

- [ ] Definire ownership per:
  - create
  - admit
  - mark prefill
  - mark decode
  - finish
  - abort
  - release cached
  - release uncached
- [ ] Spostare aggiornamenti `paged_core` nel lifecycle dove possibile.
- [ ] Spostare aggiornamenti `paged_seq_leases` nel lifecycle dove possibile.
- [ ] Spostare registrazione/invalidation prefix cache nel lifecycle dove possibile.
- [ ] Spostare integrazione lineage nel lifecycle dove possibile.
- [ ] Ridurre callback `callback_on_release` a un singolo entrypoint lifecycle.

### Criteri di completamento

- [ ] Una release paged segue una sola pipeline chiara.
- [ ] Non ci sono doppi update inconsistenti tra core, leases e cache.
- [ ] Cancel/preemption/release normale condividono percorsi prevedibili.

---

## Fase 8 — Ridurre duplicazione tra `server_slot` e `paged_request_state`

### Scopo

Eliminare duplicazione di comportamento senza forzare un modello unico prematuro.

### Task

- [ ] Confrontare campi comuni:
  - output state
  - generated tokens/probs
  - stop state
  - timings
  - sampling state
  - LoRA state
  - prompt counters
- [ ] Estrarre helper condivisi solo dove riducono duplicazione reale.
- [ ] Evitare dipendenze dal backend legacy nel paged backend.
- [ ] Consolidare `process_token()` dove possibile.
- [ ] Consolidare response final/partial dove possibile tramite callback backend-agnostiche.

### Criteri di completamento

- [ ] Meno codice duplicato in launch/token processing/response.
- [ ] Nessuna regressione legacy.
- [ ] Nessuna dipendenza paged da `server_slot` introdotta per comodità.

---

## Fase 9 — Parità funzionale e feature gap

### Scopo

Rendere esplicito e verificato cosa il paged scheduler supporta.

### Matrice iniziale

| Feature | Stato attuale stimato | Target |
| --- | --- | --- |
| Completion base | Supportata | Supportata |
| Streaming | Supportata, da verificare | Supportata |
| Cancel | Supportata, da verificare | Supportata |
| Metrics | Ibrida slot/request | Request-native |
| `/slots` save/restore/erase | Slot-centriche | Supporto reale o errore chiaro |
| Embedding | Presente nel path | Verificata |
| Rerank | Presente nel path | Verificata |
| Parent/child tasks | Presente nel path | Verificata |
| Multimodal | Presente ma delicata | Verificata o limitata |
| Speculative decoding | Disabilitata in paged | Planned o unsupported chiaro |
| Checkpoints hot path | Disabilitati | Planned o unsupported chiaro |
| Prefix cache | Supportata sperimentalmente | Ownership chiara |
| Same-seq lineage append | Env gated | Stabilizzata o mantenuta sperimentale |

### Task

- [ ] Creare test/manual repro per ogni feature.
- [ ] Aggiornare documentazione utente per feature unsupported.
- [ ] Decidere roadmap speculative decoding paged.
- [ ] Decidere roadmap checkpoints paged.

### Criteri di completamento

- [ ] Ogni feature ha uno stato documentato.
- [ ] Nessuna feature fallisce silenziosamente.
- [ ] Paged mode non promette parità dove non esiste.

---

## Fase 10 — Pulizia finale

### Scopo

Completare la separazione architetturale.

### Task

- [ ] Rimuovere branch profondi `scheduler == "paged"` da `server_context_impl`.
- [ ] Lasciare solo selezione backend in init/load.
- [ ] Eliminare uso di `slots` dal path paged.
- [ ] Rinominare funzioni generiche ancora slot-centriche, ad esempio `update_slots()` se ormai dispatcha backend.
- [ ] Pulire commenti temporanei e fallback obsoleti.
- [ ] Aggiornare docs e troubleshooting.

### Criteri di completamento

- [ ] `server_context_impl` non contiene due scheduler completi inline.
- [ ] Backend legacy e backend paged sono separati.
- [ ] Paged mode non usa `server_slot` per runtime request.
- [ ] API compatibility è confinata in adapter/layer esterno.

## Verifica per ogni fase

Eseguire almeno:

- [ ] Build server.
- [ ] Completion non-streaming legacy.
- [ ] Completion streaming legacy.
- [ ] Completion non-streaming paged.
- [ ] Completion streaming paged.
- [ ] Cancel request paged.
- [ ] Metrics paged.
- [ ] Concorrenza con più request paged.
- [ ] Pressione KV / capacity exceeded.
- [ ] Prefix cache se abilitata.
- [ ] Save/restore/erase se dichiarati supportati.

## Rischi principali

### 1. Regressioni legacy

Il codice legacy è ancora nello stesso file e condivide helper con il path paged. Ogni estrazione deve mantenere test/repro legacy.

### 2. Ownership seq-id incoerente

`paged_seq_leases`, `RequestLifecycle`, `BlockManager` e prefix cache possono divergere se una release aggiorna solo alcuni componenti.

Mitigazione:

- una sola pipeline release;
- assert/log su stato active/cached/free;
- test cancel/preemption/release normale.

### 3. API slot legacy

Le API pubbliche parlano ancora di slot. Cambiarle direttamente romperebbe client esistenti.

Mitigazione:

- compatibility adapter;
- errori espliciti in paged mode per funzioni non supportate;
- documentazione.

### 4. Feature gap nascosto

Speculative decoding e checkpoints sono disabilitati o limitati nel paged path.

Mitigazione:

- matrice feature;
- fail fast con messaggi chiari;
- non dichiarare parità finché non verificata.

### 5. File `server-context.cpp` troppo grande

Il file concentra API, routing, runtime legacy, runtime paged, metrics e lifecycle.

Mitigazione:

- estrazione incrementale;
- wrapper sottili prima, refactor profondo dopo;
- evitare riscritture massive non verificabili.

## Ordine consigliato

1. Fase 0: baseline.
2. Fase 1: backend boundary.
3. Fase 2: allocation/admission paged.
4. Fase 3: launch paged.
5. Fase 4: tick/decode paged.
6. Fase 5: handle normalization.
7. Fase 6: `/slots` e slot actions.
8. Fase 7: lifecycle ownership.
9. Fase 8: riduzione duplicazione.
10. Fase 9: feature gap/parità.
11. Fase 10: cleanup finale.

## Nota finale

La migrazione diventa pulita solo quando `server_context_impl` smette di essere il proprietario diretto di due modelli runtime. Il primo obiettivo non è cancellare `server_slot`, ma confinare il suo uso al backend legacy e impedire che il path paged lo consulti o lo simuli internamente.
