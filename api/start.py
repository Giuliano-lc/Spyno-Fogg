#!/usr/bin/env python3
"""
Script para iniciar a API suprimindo warnings do Gym.
"""

import warnings
import os
import sys

# Suprimir warnings do Gym ANTES de qualquer import
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# Agora importa e executa o main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
