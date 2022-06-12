# RadioMonteCarlo

Modello a 2 classi, label 1 (segnale) e 0 (bkg), quindi ho cambiato l'uscita della rete con un
denso a 1 uscita (invece che N_classi=2) attivato con sigmoide (invece che softmax, che vuole vettori one-hot)

Va provato usando come geometria sia x-y che r-phi

