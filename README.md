Projekat na kursu Naučno izračunavanje, Matematički fakultet, školska godina 2017/18.

Tema projekta je implementacija asinhronog one-step Q-learning algoritma. One-step Q-learning algoritam pripada familiji asinhronih deep reinforcement learning algoritama. Glavna odlika familije je mogućnost izvršavanja više instanci algoritma u okviru jednog procesora. Naravno, instance tokom svog izvršavanja međusobno komuniciraju i dele podatke što dovodi do boljih rezultata. Neophodni preduslov da bi deljeni podaci bili korisni je korišćenje asinhronog gradijentnog spusta za optimizaciju neuronske mreže. <br/>
U okviru predložene implementacije kao model je izabrana konvolutivna neuronska mreža, trenirana različitim Q-vrednostima. Ulaz u mrežu predstavljaju "sirovi" pikseli, a izlaz vrednost funkcije koja ocenjuje buduće nagrade. Algoritam je testiran na nekoliko Atari 2600 igrica korišćenjem OpenAI Gym biblioteke. U okviru direktorijuma *results* nalaze se postignuti rezultati. Pored rezultata koje je postigao implementirani algoritam nalazi se i rezultati koje su postigli konceptualno slični algoritmi iz drugih radova.


Da bi se program uspešno pokrenuo neophodno je imati instalirano:
* Python 3.6
* Tensorflow 1.8.0 (samo verzija za CPU)
* Keras	
* Skimage
* Numpy
* OpenAI Gym (posle instalacije dodati `pip install gym[atari]`)


Referentni rad koji je praćen tokom izrade nalazi se na adresi: <br/>
[link](https://arxiv.org/pdf/1602.01783v1.pdf "Asinhroni reinforcement learning")


Radovi iz kojih su uzeti rezultati nalaze se na adresama: <br/>
[link](http://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2017_2018/papers/mnih_nips_2013.pdf "Prvi rad iz koga su uzeti rezultati") <br/>
[link](https://arxiv.org/pdf/1602.01783.pdf "Drugi rad iz koga su uzeti rezultati") <br/>
[link](https://www.nature.com/articles/nature14236 "Treci rad iz koga su uzeti rezultati")
