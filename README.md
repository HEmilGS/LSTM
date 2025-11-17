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

### filtrar a 5 clases:

```bash
python3 src/filter_classes.py
```

### Entrenar el modelo

```bash
python src/train.py 
```

### Evaluar el modelo

```bash
python src/evaluate.py
```

