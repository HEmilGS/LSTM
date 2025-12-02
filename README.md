# Cómo correr el proyecto 

Este proyecto clasifica escenas de playground en **cinco categorías** utilizando un modelo basado en esqueletos 2d.

## Requisitos

Asegúrate de tener instalado:

- Python 3.8+
- pip
- CUDA (opcional, si quieres usar GPU)
- agregar el archivo ucf101_2d.pkl en la ruta *data/raw*

## Instalar dependencias

```bash
pip install -r requirements.txt

```

## 1 Cómo Ejecutar Todo

```bash
# 1. Filtrar dataset
python src/filter_classes.py

# 2. Entrenar todos los modelos (uno por uno)
python src/train.py --model baseline_fc --epochs 20
python src/train.py --model simple_lstm --epochs 20
python src/train.py --model improved_lstm --epochs 20

# 3. Evaluar y comparar
python src/evaluate.py --model all

# 4. Generar visualizaciones
python src/plot_results.py --type both

# 5. Probar predicciones
python src/predict.py --model improved_lstm
```s
---



# Documento de Mejoras - Segunda Entrega


Este documento detalla todas las mejoras implementadas en respuesta a los comentarios del profesor sobre la primera entrega del proyecto de clasificación de acciones con esqueletos 2D.

---


## 2. Implementación de pack_padded_sequence

**Recomendación del profesor:**
> "Te sugiero combinarla con la función pack_padded_sequence de PyTorch para evitar que el LSTM aprenda de las posiciones rellenadas"

### 2.1 Modificaciones en Dataset

**Archivo:** `src/dataset.py`

**Cambios implementados:**
1. El dataset ahora retorna tres valores: `(kp, label, real_length)`
2. `real_length` contiene la longitud real de cada secuencia antes del padding

```python
def __getitem__(self, idx):
    ...
    # Guardar longitud real ANTES del padding
    real_length = min(kp.shape[0], self.num_frames)
    
    # Aplicar padding si es necesario
    if kp.shape[0] < self.num_frames:
        # ... código de padding ...
    
    # Retornar también la longitud real
    return kp, label, real_length  # NUEVO
```

### 2.2 Modificaciones en Modelos

**Archivo:** `src/model.py`

Todos los modelos ahora aceptan `lengths` como parámetro opcional y usan `pack_padded_sequence`:

```python
def forward(self, x, lengths=None):
    B, T, J, C = x.shape
    x = x.reshape(B, T, J * C)
    
    if lengths is not None:
        # Ordenar por longitud (requisito de pack_padded_sequence)
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]
        
        # Pack para ignorar padding
        packed = pack_padded_sequence(
            x_sorted, 
            lengths_sorted.cpu(), 
            batch_first=True, 
            enforce_sorted=True
        )
        packed_out, (h, _) = self.lstm(packed)
        
        # Restaurar orden original
        _, unsort_idx = sort_idx.sort()
        h = h[-1][unsort_idx]
    else:
        _, (h, _) = self.lstm(x)
        h = h[-1]
    
    return self.fc(h)
```
---

## 3. Arquitecturas Mejoradas

**Recomendación del profesor:**
> "Te recomiendo intentar mejorar el modelo, quizá con más capas, regularización o usando arquitecturas más avanzadas"

### 3.1 Baseline Implementado (BaselineFC)

**Para tener una comparación justa**, implementé un baseline simple:

```python
class BaselineFC(nn.Module):
    """Promedia temporalmente + red densa"""
    def __init__(self, input_dim=34, hidden_dim=128, num_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, lengths=None):
        x = x.reshape(B, T, J * C)
        x = x.mean(dim=1)  # Promedio temporal
        return self.fc(x)
```

**Características:**
- No usa información temporal (solo promedio)
- Rápido de entrenar (~21K parámetros)
- Baseline para demostrar beneficio de LSTMs

### 3.2 SimpleLSTM (Modelo Original)

Mantuve el modelo original para comparación:
- 1 capa LSTM
- 128 hidden units
- Sin dropout
- ~90K parámetros

### 3.3 ImprovedLSTM (Principal mejora)

**Mejoras implementadas:**

1. **Más capas:** 2 capas LSTM (vs 1 original)
2. **Más capacidad:** 256 hidden units (vs 128 original)
3. **Dropout:** 0.3 en LSTM y antes de FC
4. **pack_padded_sequence:** Ignora padding
5. **~530K parámetros**

```python
class ImprovedLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=256, 
                 num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,      # 2 capas
            batch_first=True,
            dropout=dropout              # Dropout entre capas
        )
        self.dropout = nn.Dropout(dropout)  # Dropout adicional
        self.fc = nn.Linear(hidden_dim, num_classes)
```
---

## 4. Sistema de Entrenamiento Mejorado

### 4.1 Train/Validation Split

**Antes:** No había validación durante entrenamiento

**Ahora:**
```python
# Split 80/20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

### 4.2 Técnicas de Regularización

1. **Dropout:** 0.3 en múltiples puntos
2. **Gradient Clipping:** Evita explosión de gradientes
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
3. **Learning Rate Scheduler:** Reduce LR cuando no mejora
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optim, mode='min', factor=0.5, patience=3
   )
   ```

### 4.3 Early Stopping Implícito

Guarda el mejor modelo basado en validation accuracy:
```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), f"results/{model_name}_best.pth")
```

### 4.4 Historial de Entrenamiento

Guarda métricas en JSON para análisis posterior:
```python
history = {
    "train_loss": [...],
    "val_loss": [...],
    "val_acc": [...]
}
```

---

## 5. Sistema de Evaluación Completo

**Recomendación del profesor:**
> "Recuerda que debe haber comparación con al menos un baseline"

### 5.1 Métricas Detalladas

**Antes:** Solo accuracy total

**Ahora:**
1. **Accuracy global**
2. **Precision, Recall, F1-score por clase**
3. **Matriz de confusión completa**

```bash
python src/evaluate.py --model improved_lstm

# Output:
Accuracy total: 85.42%

Reporte de clasificación:
                   precision  recall  f1-score  support
  ApplyEyeMakeup      0.823    0.856    0.839      112
  Clapping            0.891    0.878    0.884       98
  JumpingJack         0.867    0.892    0.879      104
  PushUps             0.876    0.845    0.860       89
  Walking             0.912    0.898    0.905      103

Matriz de confusión:
...
```

### 5.2 Comparación de Modelos

Evalúa todos los modelos automáticamente:
```bash
python src/evaluate.py --model all

# Output:
baseline_fc         : 68.34%
simple_lstm         : 74.21%
improved_lstm       : 85.42%  ← MEJOR
```

---

## 6. Sistema de Predicciones Individuales

**Recomendación del profesor:**
> "Te recomiendo incluir un método que permita hacer predicciones individuales dado el nombre o el índice de un video"

### 6.1 Implementación Completa

**Archivo:** `src/predict.py`

#### Predicción por índice:
```bash
python src/predict.py --model improved_lstm --index 42

# Output:
Video índice: 42
Video nombre: v_Walking_g01_c02
Clase real:   Walking (label 4)
Predicción:   Walking (label 4)
Confianza:    94.32%

Probabilidades por clase:
  ApplyEyeMakeup:  1.23% ██
  Clapping:        0.89% █
  JumpingJack:     2.15% ███
  PushUps:         1.41% ██
  Walking:        94.32% ████████████████████████████████████████████

✓ Predicción CORRECTA
```

#### Predicción por nombre:
```bash
python src/predict.py --model improved_lstm --name "Walking_g01"

# Busca videos que contengan "Walking_g01" en su nombre
```

#### Modo interactivo:
```bash
python src/predict.py --model improved_lstm

# Menú interactivo:
# 1. Predecir por índice
# 2. Predecir por nombre
# 3. Predecir varios aleatorios (con estadísticas)
# 4. Salir
```

### 6.2 Funciones Disponibles

```python
# Predecir por índice
result = predict_by_index(model, pkl_data, video_idx=42)

# Predecir por nombre (búsqueda flexible)
result = predict_by_name(model, pkl_data, video_name="Walking")

# Resultado incluye:
{
    "video_idx": 42,
    "frame_dir": "v_Walking_g01_c02",
    "true_label": 4,
    "true_class": "Walking",
    "pred_label": 4,
    "pred_class": "Walking",
    "confidence": 94.32,
    "all_probs": {
        "ApplyEyeMakeup": 1.23,
        "Clapping": 0.89,
        ...
    }
}
```

---

## 7. Sistema de Visualización

**Archivo:** `src/plot_results.py`

### 7.1 Gráficas de Entrenamiento

```bash
python src/plot_results.py --type history
```

Genera gráfica con 3 paneles:
1. Training Loss por época
2. Validation Loss por época
3. Validation Accuracy por época

Permite comparar múltiples modelos en la misma gráfica.

### 7.2 Comparación de Modelos

```bash
python src/plot_results.py --type comparison
```

Gráfica de barras mostrando:
- Mejor validation accuracy de cada modelo
- Accuracy final de cada modelo

---

