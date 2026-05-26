dgxStress.py:
-------------
1) Tensor Cores: L'uso di torch.float16 (Half Precision) è fatto appositamente 
per attivare i Tensor Cores, che sono i componenti della GPU che consumano più 
energia e generano più calore in assoluto sulle DGX.

2) Saturazione VRAM: Se la DGX ha schede con molta memoria (es. 40GB, 80GB o più 
per GPU), potresti voler aumentare il valore di BATCH_SIZE (es. BATCH_SIZE = 12) 
o MATRIX_SIZE (es. 24576) finché l'utilizzo della memoria mostrato da nvidia-smi 
non sfiora il 90-95%. Attenzione a non esagerare per evitare l'errore di Out of 
Memory (OOM).

3) Multiprocessing: Lo script usa il modulo multiprocessing di Python. 
Questo è fondamentale sulle DGX per bypassare il GIL (Global Interpreter Lock) 
di Python e garantire che ogni GPU riceva comandi alla massima velocità possibile 
da un core CPU dedicato.


Calcolo dell'impatto sulla VRAM:
- Una matrice di dimensioni 32768 * 32768 contiene circa 1,07 miliardi di elementi.
- In precisione torch.float16 (FP16), ogni elemento occupa 2 byte. 
Quindi, una singola matrice occupa circa 2,15 GB di VRAM.
- Nel ciclo allochiamo matrices_A e matrices_B, quindi 24 + 24 = 48 matrici 
in totale per GPU.
- 48 matrici *  2,15GB approx 103,2GB di VRAM occupata solo per i dati di partenza.
- Il resto della VRAM (fino a circa 115-120 GB) verrà dinamicamente saturato dai 
tensori temporanei generati dalle operazioni di moltiplicazione (torch.matmul) 
e dai contesti CUDA.

************************************************************************************
dgxStressLLM.py
---------------
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
