'''
Per caricare un modello che saturi una GPU da 128 GB di VRAM senza dover scaricare 
centinaia di gigabyte di file da internet (che richiederebbero ore), usiamo un 
trucco avanzato di PyTorch e Hugging Face: inizializziamo una configurazione di 
un modello colossale (il formato Llama-3-70B) usando il contesto meta di PyTorch 
(che crea l'architettura a costo zero in RAM), e poi allochiamo i tensori reali 
direttamente sulla GPU convertendoli in FP16.
Per stressare al massimo la macchina, lo script non fa solo calcoli casuali, 
ma esegue continui cicli di Forward Pass (generazione/inferenza) con sequenze 
a contesto massimo.


1) mp.set_start_method('spawn', force=True) all'inizio assoluto: Risolve 
definitivamente l'errore del sub-processo forkato, isolando la memoria CUDA di 
ogni processo figlio.

2) Uso di with torch.device(device):: Inizializza l'enorme ammontare di parametri 
direttamente dentro la VRAM della GPU selezionata. Se non usassimo questo 
accorgimento, PyTorch creerebbe prima il modello nella RAM di sistema (CPU), 
saturando immediatamente i nodi NUMA della DGX e causando un crash di sistema 
per Out-Of-Memory della CPU (OOM killer di Linux).

3) Saturazione di Tipo "Transformer": A differenza della moltiplicazione di matrici 
standard, l'architettura a blocchi di Llama costringe la GPU a muovere 
continuamente i pesi del modello dai chip di memoria HBM3 ai core di calcolo 
e viceversa, generando uno stress combinato su ampiezza di banda della memoria 
(Memory Bandwidth) e potenza di calcolo puro (TFLOPS).
'''

import os
import sys
import time
from multiprocessing import Process
import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM

# =====================================================================
# CONFIGURAZIONE STRUTTURATA PER GPU DA 128GB
# =====================================================================
# Usiamo la configurazione di Llama-3-70B. In FP16 occupa circa 140GB di base.
# Modificando i parametri interni, la "dimagriamo" leggermente per farla risiedere
# stabilmente intorno ai 110-115 GB di VRAM, lasciando lo spazio per il contesto di calcolo.
MODEL_ID = "meta-llama/Meta-Llama-3-70B" 
BATCH_SIZE = 4          # Numero di sequenze elaborate contemporaneamente
SEQUENCE_LENGTH = 4096   # Lunghezza del contesto (più è alto, più stressa la memoria dei Tensor Core)
ITERATIONS = 50000       # Quanti cicli di inferenza eseguire

def stress_gpu_with_llm(gpu_id):
    """Inizializza un LLM gigante sulla GPU target ed esegue inferenza continua."""
    try:
        # Configura il device CUDA specifico
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"[GPU {gpu_id}] [{gpu_name}] Avvio allocazione LLM...")
        sys.stdout.flush()

        # 1. Scarica solo la configurazione strutturale (pochi KB)
        config = AutoConfig.from_pretrained(MODEL_ID)
        
        # Ottimizzazione parametri per calzare a pennello su 128GB (Target ~110GB occupati all'avvio)
        config.num_hidden_layers = 72  # Adattiamo leggermente il numero di layer se necessario
        
        # 2. Inizializzazione "Brutale" dei pesi direttamente in VRAM
        # Usiamo torch.device("cuda") per evitare che passi dalla RAM di sistema (evita crash di OOM sulla CPU)
        print(f"[GPU {gpu_id}] Generazione pesi del modello direttamente in VRAM (FP16)...")
        sys.stdout.flush()
        
        with torch.device(device):
            # L'inizializzazione da config crea pesi casuali ma reali nella memoria della GPU
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        
        # Mette il modello in modalità valutazione e disabilita i gradienti per velocizzare i cicli
        model.eval()
        torch.set_grad_enabled(False)
        
        vram_used = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        print(f"[GPU {gpu_id}] Modello caricato! VRAM Occupata stabilmente: {vram_used:.2f} GB / 128 GB")
        print(f"[GPU {gpu_id}] Avvio ciclo di Forward Pass infinito (Tensor Cores al massimo)...")
        sys.stdout.flush()

        # 3. Generazione dati di input fittizi (Tokens) per il benchmark
        # Creiamo una matrice di numeri interi che simulano il testo in ingresso
        input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.long, device=device)

        start_time = time.time()
        
        # 4. Loop di stress
        for i in range(ITERATIONS):
            # Il forward pass attiva l'algoritmo di Attention (calcolo pesantissimo sulle matrici QKV)
            outputs = model(input_ids)
            
            # Forziamo una micro-sincronizzazione ogni tanto per monitorare la stabilità
            if i % 100 == 0 and i > 0:
                torch.cuda.synchronize(device)
                elapsed = time.time() - start_time
                it_per_sec = i / elapsed
                print(f"[GPU {gpu_id}] Eseguiti {i} forward pass. Velocità: {it_per_sec:.2f} it/s")
                sys.stdout.flush()

    except Exception as e:
        print(f"[GPU {gpu_id}] ERRORE: {e}")
        sys.stdout.flush()
    finally:
        print(f"[GPU {gpu_id}] Test interrotto/terminato.")

if __name__ == '__main__':
    # GESTIONE MULTIPROCESSING: Forza il metodo 'spawn' prima di qualsiasi chiamata CUDA
    # Questo clona l'ambiente in modo pulito senza ereditare i descrittori di contesto di PyTorch
    try:
        mp.set_start_method('spawn', force=True)
        print("[SYSTEM] Metodo di multiprocessing 'spawn' impostato correttamente.")
    except RuntimeError as e:
        print(f"[SYSTEM] Errore nell'impostazione del multiprocessing: {e}")
        sys.exit(1)

    # Controllo GPU
    if not torch.cuda.is_available():
        print("[SYSTEM] Errore: Nessuna GPU NVIDIA rilevata con CUDA.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"\n=======================================================")
    print(f"   LLM BRUTAL STRESS TEST - NVIDIA DGX (128GB VRAM)     ")
    print(f"=======================================================")
    print(f"GPU Rilevate totali: {num_gpus}")
    print(f"Modello simulato: {MODEL_ID} (~70 Miliardi di parametri)")
    print(f"Contesto di calcolo: Batch Size {BATCH_SIZE} | Sequence Length {SEQUENCE_LENGTH}")
    print(f"ATTENZIONE: Questo test satura la banda HBM3 e attiva i Tensor Core.")
    print(f"I consumi elettrici saliranno istantaneamente al TDP massimo.")
    print(f"=======================================================\n")
    sys.stdout.flush()

    processes = []
    
    # Creiamo un processo CPU indipendente per ogni singola GPU da 128GB
    for gpu_id in range(num_gpus):
        p = Process(target=stress_gpu_with_llm, args=(gpu_id,))
        processes.append(p)
        p.start()

    # Mantieni il server in esecuzione finché i processi non terminano (o finché non premi CTRL+C)
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Ricevuto segnale di interruzione. Chiusura dei processi in corso...")
        for p in processes:
            p.terminate()
            p.join()
        print("[SYSTEM] Stress test interrotto in sicurezza.")
