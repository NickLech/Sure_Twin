# Scelta della configurazione: horizon 10 min, finestra 9-18 h

**Configurazione scelta: horizon = 10 min, finestra = 9-18 h** (90% in finestra, 75% fuori finestra), approccio *scheduled, calibrated per period*.

## Criterio

Per ogni fascia oraria (finestra e altre ore) si cerca la barra più bassa su due metriche:

- **Unmet traffic (%)**: traffico non coperto (più basso = più sicuro)
- **Over-provisioning (%)**: spreco di risorse (più basso = meno spreco)

La colonna "all hours" è ignorata. Valori calcolati dal test set (`schedule_final_metrics_all.csv`), approccio *scheduled*.

## Risultati

| Horizon | Finestra | Unmet finestra | Unmet altre ore | Over-prov. finestra | Over-prov. altre ore |
|---|---|---|---|---|---|
| **10 min** | **9-18** | **1.06** | **2.62** | **16.60** | 9.08 |
| 10 min | 9-19 | 1.06 | 2.65 | 16.72 | 9.09 |
| 10 min | 8-21 | 1.08 | 2.65 | 17.14 | 9.01 |
| 20 min | 9-18 | 1.14 | 2.83 | 18.09 | 9.76 |
| 20 min | 9-19 | 1.13 | 2.86 | 18.19 | 9.77 |
| 20 min | 8-21 | 1.16 | 2.88 | 18.60 | 9.66 |
| 30 min | 9-18 | 1.17 | 2.90 | 18.77 | 10.17 |
| 30 min | 9-19 | 1.17 | 2.94 | 18.85 | 10.20 |
| 30 min | 8-21 | 1.19 | 2.96 | 19.27 | 10.09 |

## Perché horizon 10 min

A parità di finestra, passare da 10 a 20 o 30 min peggiora **tutte e quattro** le metriche (con 9-18: unmet in finestra 1.06 → 1.14 → 1.17, over-prov. in finestra 16.60 → 18.09 → 18.77). Un orizzonte più corto dà previsioni più affidabili, quindi non c'è alcun trade-off: 10 min è migliore sia per copertura sia per spreco.

## Perché finestra 9-18

Con horizon 10 min le tre finestre sono molto vicine, ma 9-18 è la migliore o alla pari su 3 metriche su 4:

- **Unmet in finestra**: 1.06, pari a 9-19 (8-21: 1.08)
- **Unmet altre ore**: 2.62, il più basso (9-19 e 8-21: 2.65)
- **Over-prov. in finestra**: 16.60, il più basso (9-19: 16.72, 8-21: 17.14)
- **Over-prov. altre ore**: 9.08, praticamente identico alle altre (9.01-9.09), differenza trascurabile

## Limiti

Le differenze tra le tre finestre sono piccole (≤ 0.5 punti percentuali in over-prov., ≤ 0.03 in unmet). La scelta di 9-18 è quindi di ottimizzazione marginale; la scelta di horizon 10 min è invece netta.
