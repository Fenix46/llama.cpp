# Paged KV Cache — Intervention Points

> Branch: `paged-kv-cache-phase1`
> Scope: analysis only, no code modified.

---

## 1. Struttura dati KV cache attuale

### Gerarchia dei tipi

```
llama_kv_cache_iswa          (src/llama-kv-cache-iswa.h)
  ├─ llama_kv_cache  kv_base     // layer senza SWA
  └─ llama_kv_cache  kv_swa      // layer con SWA

llama_kv_cache               (src/llama-kv-cache.h:20)
  ├─ vector<llama_kv_cells>  v_cells      // [n_stream] metadata celle
  ├─ vector<uint32_t>        v_heads      // [n_stream] ring-buffer head
  ├─ vector<uint32_t>        seq_to_stream // seq_id → stream_id
  ├─ uint32_t  n_stream      // 1=unified, n_seq_max=per-seq
  ├─ uint32_t  n_swa         // dimensione finestra scorrevole
  ├─ vector<Layer>  layers   // per-layer K/V tensors
  │    Layer.k_stream[s]     // ggml_tensor* K del layer l, stream s
  │    Layer.v_stream[s]     // ggml_tensor* V del layer l, stream s
  └─ stream_copy_info  sc_info  // buffer copy in sospeso

llama_kv_cells               (src/llama-kv-cells.h:32)
  ├─ vector<llama_pos>       pos      // pos[i] = posizione cella i (-1=vuota)
  ├─ vector<llama_kv_cell_ext> ext    // ext[i] = coords 2D (M-RoPE)
  ├─ vector<llama_pos>       shift    // shift accumulato (RoPE)
  ├─ vector<bitset<256>>     seq      // seq[i] = bitset seq_id occupanti
  ├─ map<llama_pos,int>      seq_pos[256]  // seq_pos[s][p] = count → O(log n) min/max
  └─ set<uint32_t>           used     // indici celle non-vuote
```

### Tensori K/V per layer

```
K: shape  [n_embd_head_k, n_head_kv, kv_size, n_stream]
V: shape  [n_embd_head_v, n_head_kv, kv_size, n_stream]  (non-transposed / FA)
   shape  [kv_size, n_head_kv, n_embd_head_v, n_stream]  (transposed / no-FA)
```

`kv_size` = numero totale celle pre-allocate al momento della creazione del context. Non cambia a runtime.

---

## 2. Come vengono assegnate le posizioni KV

### Pipeline completa

```
llama_decode(ctx, batch)                        [llama-context.cpp:1533]
  │
  ├─ balloc->init(batch)                        // sanitize batch, auto-fill pos
  ├─ memory->init_batch(balloc, n_ubatch, ...)  // split in ubatch, chiama prepare()
  │    └─ prepare()                             [llama-kv-cache.cpp:675]
  │         └─ find_slot(ubatch, cont=false)    [llama-kv-cache.cpp:817]
  │              → slot_info {idxs[seq][tok] = cell_index}
  │
  └─ per ogni ubatch:
       apply_ubatch(sinfo, ubatch)              [llama-kv-cache.cpp:1016]
         // commit: pos_set, seq_add, avanza v_heads
       build_graph(ubatch, sinfo)
         ├─ cpy_k  → ggml_set_rows(k, k_cur, k_idxs)
         └─ cpy_v  → ggml_set_rows(v, v_cur, v_idxs)
```

### `find_slot()` — ring-buffer scan

**Firma:** `slot_info find_slot(const llama_ubatch & ubatch, bool cont) const`  
**Chiamato sempre con** `cont=false` → non richiede celle contigue.

```
head_cur = v_heads[stream]
if head_cur > used + 2*n_tokens: head_cur = 0   // ottimizzazione

for ogni token da piazzare:
    idx = head_cur++ (con wraparound a cells.size())
    
    accetta idx se:
      - cells.is_empty(idx)                      // cella libera
      - oppure: 1 sola seq E is_masked_swa(...)  // fuori finestra SWA

ritorna slot_info con idxs[seq][tok] = idx
```

### `apply_ubatch()` — commit

```
per ogni token t in ubatch:
    cells.pos_set(idxs[s][t], pos[t])
    cells.seq_add(idxs[s][t], seq_id)
    se M-RoPE: cells.ext_set(...)
    
    se SWA: purga pos ≤ pos_overwritten (invariante contiguità)
v_heads[s] = next_head
```

---

## 3. Come funzionano `seq_id` e `pos`

### `llama_pos` — `int32_t`
- Posizione del token nella sequenza (0, 1, 2, …)
- Auto-generata da `balloc->init()` se non fornita
- Una cella può avere **una sola `pos`**, ma più `seq_id`
- Usata come indice per K-shift RoPE, SWA masking, min/max query

### `llama_seq_id` — `int32_t`
- Identifica la sequenza (es. request parallela, beam)
- Un token può appartenere a **più seq_id** (`n_seq_id[t] > 1`)
- Limite: `LLAMA_MAX_SEQ = 256`
- Una cella tiene i propri seq_id in un `bitset<256>` → test O(1)
- `seq_pos[s]` = mappa `pos→count` per seq_id `s` → min/max O(log n)

### Invariante cella
```
cells.pos[i]  = p     → cella i contiene dato per posizione p
cells.seq[i]  = {a,b} → le seq a e b usano questa cella
```
Più seq con stessa pos condividono la cella (es. dopo `seq_cp`).

### Operazioni

| Funzione | Riga | Effetto |
|----------|------|---------|
| `seq_rm(s, p0, p1)` | 342 | rimuove seq s da [p0,p1), libera celle orfane |
| `seq_cp(src, dst, p0, p1)` | 405 | stesso stream: bitset add. Cross-stream: enqueue buffer copy |
| `seq_keep(s)` | 492 | evict tutte le altre seq da tutte le celle |
| `seq_add(s, p0, p1, δ)` | 514 | shift pos → K-shift RoPE graph |
| `seq_div(s, p0, p1, d)` | 559 | scala pos → compressione |

---

## 4. Dove avviene la ricerca slot

### Stack completo con file:righe

```
llama_decode                          llama-context.cpp:1533
  memory->init_batch()                llama-kv-cache.cpp (via llama_memory_i)
    prepare()                         llama-kv-cache.cpp:675
      find_slot(ubatch, cont=false)   llama-kv-cache.cpp:817
        v_cells[stream]               (ring-buffer su llama_kv_cells)
```

`prepare()` è non-distruttiva: applica tentatively, poi rollback. Solo dopo, `apply_ubatch()` committer.

### `n_kv` — dimensione KV passata all'attention

```cpp
// llama-kv-cache.cpp:1128
uint32_t get_n_kv(const slot_info & sinfo) const {
    result = GGML_PAD(cells.used_max_p1(), n_pad_cur)
    // padded a multiplo di 256, mai più di cells.size()
}
```

Non è sempre `cells.size()`: è `max_used_index + 1` arrotondato. L'attention vede **solo le celle in [0, n_kv)**.

---

## 5. Parti che assumono KV contigua

### ✅ Runtime decode path — NON assume contiguità

**Tutti i path critici usano gather/scatter:**

| Operazione | Meccanismo | File:riga |
|-----------|-----------|-----------|
| `cpy_k()` | `ggml_set_rows(k, k_cur, k_idxs)` | kv-cache.cpp:1229 |
| `cpy_v()` | `ggml_set_rows(v, v_cur, v_idxs)` | kv-cache.cpp:1285 |
| `build_input_k_idxs()` | tensor I64 `[n_tokens]` con indici cella arbitrari | kv-cache.cpp:1287 |
| `build_input_v_idxs()` | tensor I64 `[n_tokens]` o `[n_tokens*n_embd_v_gqa]` | kv-cache.cpp:1297 |
| `set_input_kq_mask()` | itera su 0..n_kv, usa `cells.is_empty(j)` | kv-cache.cpp:1608 |
| `get_k()` / `get_v()` | `ggml_view_4d` con stride regolari su slab piatto | kv-cache.cpp:1144 |

### ⚠️ State save/restore — OTTIMIZZAZIONE contigua

`is_contiguous()` usata in serializzazione KV state:

```cpp
// kv-cache.cpp:2237, 2281, 2333
if (sinfo.is_contiguous()) {
    // fast path: singola memcpy
    ggml_backend_tensor_set(k, src, head*row_size, count*row_size);
} else {
    // slow path: scatter per-cell
    for (i in 0..count):
        ggml_backend_tensor_set(k, src+i*row, idxs[0][i]*row, row_size);
}
```

**Non è un requisito**, è un'ottimizzazione. La slow path funziona con qualsiasi layout.

### ⚠️ `find_slot()` con `cont=true` — esiste ma non usato

Il parametro `cont` esiste nel codice (riga 817) ma viene **sempre chiamato `false`**. Se in futuro si usasse `cont=true`, richiederebbe celle contigue. Oggi: dead path.

### ⚠️ `n_kv` come indice diretto nel mask

`set_input_kq_mask` itera `for j in 0..n_kv` e usa `j` direttamente come **indice fisico** nella cella. Non è un indice logico. Il mask è indicizzato per posizione fisica nella slab. Con una page table, questo loop cambierebbe.

---

## 6. Funzioni impattate da KV a blocchi (paged KV)

### Impatto diretto — devono cambiare

| Funzione | File | Perché |
|----------|------|--------|
| `find_slot()` | kv-cache.cpp:817 | Sostituire ring-buffer con page-table lookup |
| `apply_ubatch()` | kv-cache.cpp:1016 | Allocare pagine invece di singole celle |
| `get_n_kv()` | kv-cache.cpp:1128 | `n_kv` diventa `n_pages * page_size`, non `used_max+1` |
| `get_k()` / `get_v()` | kv-cache.cpp:1144 | View su slab fisico potrebbe non coprire pagine sparse |
| `build_input_k_idxs()` | kv-cache.cpp:1287 | `idxs` già arbitrari → mappa tramite page table |
| `set_input_k_idxs()` | kv-cache.cpp:1354 | Calcolo offset: `page_id * page_size + intra_page_offset` |
| `set_input_kq_mask()` | kv-cache.cpp:1608 | Loop su `n_kv` fisici → loop su pagine logiche |
| `seq_rm()` | kv-cache.cpp:342 | Liberare pagine intere quando vuote |
| `seq_cp()` | kv-cache.cpp:405 | Copiare page table entry, non singole celle |

### Impatto indiretto — interfaccia stabile ma semantica cambia

| Funzione | File | Nota |
|----------|------|------|
| `prepare()` | kv-cache.cpp:675 | Logica identica, chiama `find_slot()` modificato |
| `update()` | kv-cache.cpp:741 | K-shift graph invariato; buffer copy cambia se pagine non contigue |
| `llama_kv_cells` | kv-cells.h:32 | Aggiungere `page_id` field; o wrappare in `llama_kv_page` |
| `llama_kv_cache` constructor | kv-cache.cpp | Allocazione: N pagine fisse → pool di blocchi |

### Nessun impatto — stabili

| Layer | Perché |
|-------|--------|
| `llama_memory_i` interface | Astrazione già presente; paging è implementazione |
| `llama_decode()` / `process_ubatch()` | Usa interfaccia `mctx`, non tocca cache direttamente |
| Operazioni seq pubbliche (`seq_rm`, `seq_add`, …) | API stabile; implementazione interna cambia |
| Server (`server_slot`) | Usa solo `slot.id` come `seq_id`; nessun riferimento a celle fisiche |
| `llama_batch` / `llama_ubatch` | Strutture dati input; non dipendono da layout fisico |
| GGML ops (`ggml_flash_attn_ext`) | Riceve K/V come view/tensor; non conosce il layout cella |

---

## 7. Server layer — rapporto con KV cache

`server_slot` (server-context.cpp:79):

```cpp
struct server_slot {
    int id;          // usato come seq_id in tutte le API llama_memory_*
    int32_t n_ctx;   // context size per slot
    // NO kv_slot, NO indici fisici
};
```

Il server usa **solo seq_id**:
- `llama_memory_seq_rm(ctx, slot.id, p0, p1)`
- `llama_memory_seq_cp(ctx, id, other.id, -1, -1)`
- `llama_memory_seq_add(ctx, slot.id, n_keep+n_discard, size, -n_discard)`

Il server non sa nulla del layout fisico → **paging trasparente** rispetto al server layer.

`n_past` locale (server-context.cpp:2292) = contatore software di tokens processati, non un indice KV.

---

## 8. Schema intervento Phase 2

```
Nuovo struct: llama_kv_page_table
  map<(seq_id, page_idx), physical_block_id>
  physical_block_id → indice nel pool di blocchi

Modifiche minime:
1. llama_kv_cells: aggiunge concetto di page boundary
2. find_slot(): invece di scan ring-buffer per singole celle,
                alloca page intera (block_size celle contigue nel slab)
3. set_input_k_idxs(): per token t in pagina p, posizione intra-page i:
   index = blocks[page_table[seq][p]] * block_size + i
4. seq_rm(): libera page quando tutti i cells della page sono vuoti
5. get_n_kv(): arrotonda a block_size multiplo
```

**Invariante da preservare:** `build_input_k_idxs()` già produce indici fisici arbitrari. Con paging, produce `physical_block * block_size + intra`. Nessun cambio a GGML né al compute graph.
