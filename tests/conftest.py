"""Configuración común de tests.

Limitar hilos de CPU: con 24 hilos OpenMP y otros procesos activos (otro entrenamiento en la máquina),
la sobresuscripción hacía que construir un modelo pequeño tardara ~47 s en vez de ~0.2 s.
"""

import torch

torch.set_num_threads(2)
