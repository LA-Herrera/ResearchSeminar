Superresolución de Imágenes de Vigilancia

Red neuronal convolucional ligera para la superresolución de imágenes individuales, destinada a imágenes de vigilancia y circuito cerrado de televisión (CCTV). El modelo combina un módulo inicial de ampliación mediante interpolación múltiple con una pila secuencial de refinamiento mediante NAFBlocks.

Elaborado en el marco de un proyecto de seminario de investigación.

---

## Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales: `torch`, `torchvision`, `basicsr`, `opencv-python`,
`numpy`, `scipy`, `tqdm`, `piq`, `openpyxl`.

---

## Estructura del repositorio

```
├── Dataset/
|   ├── HR/
|   ├── LR/
|   └── synthetic_degradation.py  # Proceso de degradación sintética
├── Experiments/
|   ├── results/                  # Se crea automaticamente durante las pruebas
|   └── models/
├── Options/
│   ├── train/                    # Configuraciones YAML de entrenamiento
│   └── test/                     # Configuraciones YAML de pruebas
├── Training/
|   ├── archs/
|   |   └── vn_arch.py            # Arquitectura del modelo
|   ├── models/
|   |   └── vn_model.py           # Logica de entrenamiento
|   └── utils/
|   |   ├── loss.py               # FFT, Gradiente, Varianza Local
|   |   └── metrics.py            # LPIP, DISTS
├── requirements.txt
├── test.py
└── train.py
```

---

## Entrenamiento

```bash
python train.py -opt Options/train/vn_train.yml
```

Para cargar los parametros de una sesión anterior sin reanudar el estado completo del entrenamiento (ej. al cambiar el número de bloques entre experimentos):

```yaml
# En el archivo YAML
path:
  pretrain_network_g: Experiments/models/net_g_140600.pth
  strict_load_g: false
  resume_state: ~
```

---

## Pruebas

```bash
python test.py -opt Options/test/vn_test.yml
```

Los resultados son guardados en  `Experiments/results/`.

---

## Notas de configuración

Toda la configuración del entrenamiento se controla a través de los archivos YAML en `Options/train/`. Cada experimento debe utilizar un campo `name` distinto para mantener separados los puntos de control y los registros:

```yaml
name: VN_x4_d8
```

Las combinaciones de pérdidas se definen en la sección `train` del archivo YAML. Todas las configuraciones utilizan Charbonnier como pérdida de píxel base. Los términos adicionales se añaden según sea necesario con sus respectivos pesos.

---

## Notas

- Entrenado y evaluado con imágenes de vigilancia de TJU-DHD
- El proceso de degradación sintética sigue un diseño de dos pasadas inspirado en BSRGAN y Real-ESRGAN
- Metricas de Evaluación: PSNR, SSIM, LPIPS, DISTS
