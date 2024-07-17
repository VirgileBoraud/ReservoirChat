C’est l’histoire d’un GAN
Marie-Agnès Enard, Pascal Guitton, Thierry Viéville

To cite this version:

Marie-Agnès Enard, Pascal Guitton, Thierry Viéville. C’est l’histoire d’un GAN. Blog binaire - Le
Monde, 2020. ￿hal-02937739￿

HAL Id: hal-02937739

https://inria.hal.science/hal-02937739

Submitted on 14 Sep 2020

HAL is a multi-disciplinary open access
archive for the deposit and dissemination of sci-
entific research documents, whether they are pub-
lished or not. The documents may come from
teaching and research institutions in France or
abroad, or from public or private research centers.

L’archive ouverte pluridisciplinaire HAL, est
destinée au dépôt et à la diffusion de documents
scientifiques de niveau recherche, publiés ou non,
émanant des établissements d’enseignement et de
recherche français ou étrangers, des laboratoires
publics ou privés.

Distributed under a Creative Commons Attribution 4.0 International License

PUBLIÉ LE
14 SEPTEMBRE 2020 PAR BINAIRE

C’est l’histoire d’un GAN

https://www.lemonde.fr/blog/binaire/2020/09/14/cest-lhistoire-dun-gan 

Oui binaire s’adresse aussi aux jeunes de tous âges que le 
numérique laisse parfois perplexes. Avec « Petit binaire », osons 
ici expliquer de manière simple et accessible un des mécanismes 
qui nous donne l’impression d’une intelligence artificielle : les 
réseaux antagonistes génératifs (Generative Adversial Network 
ou GAN en anglais). Marie-Agnès Enard, Pascal Guitton et 
Thierry Viéville.

[caption id="" align="aligncenter" width="411"]

Belamy ©Public Domain[/caption]

 Le Portrait d' Edmond de 

432 500 dollars une fois … 432 500 dollars deux fois … 432 500 dollars trois fois … adjugé ! 
Nous sommes chez Christie's le 25 octobre 2018, ce “Portrait d'Edmond de Belamy”, une 
toile d’une série représentant une famille bourgeoise fictive des XVIIIe et XIXe siècle, vient 
de partir à un bon prix. Et tu sais quoi ? C’est la première œuvre d'art, produite par un 
logiciel d'intelligence artificielle, à être présentée dans une salle des ventes.

- Ah l’arnaque !!!

Ah non non, ce n’est pas une escroquerie, les gens le savaient, regarde la signature : c’est 
une formule mathématique en référence au code de l'algorithme utilisé.

 
 
[caption id="" align="alignright" width="212"]
par le réseau adverse génératif StyleGAN, en se basant sur une analyse de portraits. 
L'image ressemble fortement à une photo d'une vraie personne. ©OwlsMcGee [/caption]

 Image générée 

- Oh … mais c’est dingue, ça marche comment ton algorithme ?

C’est assez simple, deux réseaux de calcul sont placés en compétition :

Le premier réseau est le “générateur”, à partir d’exemples (des tableaux du XIXe siècle), il 
génère un autre objet numérique artificiel (un tableau inédit “dans le style” du XIXe siècle);

Son adversaire le “discriminateur” essaie de détecter si le résultat est réel ou bien s'il est 
généré.

- Et alors ?

Le générateur adapte petit à petit ces paramètres pour maximiser les chances de tromper le 
discriminateur qui lui aussi adapte ces paramètres pour maximiser les chances de 
démasquer le générateur. Bref le second entraîne le premier qui finit par être super 
performant, comme cette image qui ressemble fortement à une photographie d'une vraie 
personne.

[caption id="attachment\_11405" align="aligncenter" width="990"]

Architecture d’un réseau antagoniste génératif (generative adversial network ou GAN en 
anglais) : on voit le générateur créer des données factices dont la diversité provient d’un 

 
générateur aléatoire, et un discriminateur recevoir (sans savoir quoi est quoi) un mélange de 
données réelles et factices, à chacune de ces prédictions l’erreur est répercutée pour 
permettre d’ajuster les paramètres : pour le discriminateur l’erreur est de ne pas distinguer le 
réel du factice et pour le générateur de se faire démasquer.
©geeksforgeeks[/caption]

- Attends … tu peux être plus clair et revenir en arrière : c’est quoi un “réseau de calcul” ?

Ah oui pardon. Très simplement on s’est aperçu que si on cumule des millions de calculs en 
réseaux (on parle de réseaux de neurones) on peut approximativement faire tous les calculs 
imaginables, comme reconnaître un visage, transformer une phrase dictée en texte, etc… Et 
pour apprendre le calcul à faire, on fournit énormément d’exemples et un mécanisme très 
long ajuste les paramètres du calcul en réduisant pas à pas les erreurs pour avoir de bonnes 
chances d’obtenir ce que l’on souhaite.

- C’est tout ?

Dans les grandes lignes, oui, on l’explique ici, c’est un truc rigolo et tu peux même jouer 
avec ici si tu veux.

https://www.lemonde.fr/blog/binaire/2017/10/20/jouez-avec-les-neurones-de-la-machine

[caption id="attachment\_11406" align="aligncenter" width="1004"]

Vue de l’interface qui permet de “jouer avec des neurones artificiels” : on voit en entrée les 
caractéristiques calculées sur l’image (features) par exemple X1 discrimine la gauche de la 
droite (envoie une valeur négative si le point est à gauche et positive si il est à droite), puis 
les unités de calculs (les “neurones”) organisées en couche (layers) jusqu’à la sortie du 
calcul. ©playground.tensorflow.org[/caption]

- J’ai bien entendu dans tes explications que tu as dit “on peut approximativement faire” et 
“on a de bonnes chances” ! Donc c’est pas exact et en plus ça peut toujours se tromper ??

 
[caption id="attachment\_11407" align="alignleft" width="193"]
Un urinoir en porcelaine renversé et signé : la création artistique probablement la plus 
controversée de l'art du XXIe siècle. ©gnu[/caption]

Absolument, c’est pour cela qu’il faut bien comprendre les limites de ces mécanismes qui 
donnent des résultats bluffants mais restent simplement de gros calculs. Mais bon ici on s’en 
sert pour faire de l’art donc obtenir un résultat inattendu peut même se révéler intéressant.

- Tu veux dire que le calcul est “créatif” car il peut se tromper et faire n’importe quoi ?

Oui mais pas uniquement. La créativité correspond bien au fait de ne pas faire ce qui est 
attendu, mais il faut un deuxième processus qui “reconnaît” si le résultat est intéressant ou 
pas.

Pour parler de création artistique il faut être face à quelque chose de :

1/ singulier, original donc, résultat d’une création, par opposition à une production usuelle ou 
reproduction

ET

2/ esthétique, s’adressant délibérément aux sens (vision, audition), à nos émotions, notre 
intuition et notre intellect (nous faire rêver, partager un message), par opposition à une 
production utilitaire.

- Ah oui mais alors quelque chose ne peut être reconnu comme artistique que par un 
humain ?

Oui, ou par un calcul dont les paramètres ont été ajustés à partir d’un grand nombre de 
jugements humains.

- Et ces calculs que tu appelles “réseaux antagonistes génératifs”, ils servent juste à faire de 
l’art ?

Pas uniquement, on les utilisent dans d’autres formes artistiques comme la musique aussi, 
mais au delà pour générer des calculs qui explorent des solutions à des problèmes 
complexes comme par exemple : pour la découverte de nouvelles structures moléculaires 
(encore au stade des essais), en astronomie pour modéliser des phénomènes complexes, 
et sous une forme un peu différente pour gagner au jeu de go. L’intérêt est de fournir des 
nouvelles données d'entraînement (tu sais qu’il en faut des millions) quand elles sont à la 
base en nombre insuffisant.

- Tu sais ce qui me surprend le plus ? C’est que c’est finalement facile à comprendre.

Cool :)

[caption id="attachment\_11408" align="alignright" width="238"]

 
 
artistique sur la création artistique par Jacques Prévert. ©scribd[/caption]

 Pour faire le portrait d’un oiseau. Une réflexion 

- N’empêche que ça questionne sur ce que l’on peut considérer comme de l’art, in fine.

Oui ce qu’on appelle l’intelligence artificielle, remet surtout en question ce que nous 
considérons comme naturellement intelligent.

- Et tu crois que “ça” pourrait faire de la poésie ?

De la mauvaise poésie peut-être… laissons le dernier mot à Jacques Prévert :

[caption id="attachment\_11409" align="alignleft" width="202"]
©TîetRî[/caption]

« il se croyait pouet

car il sonnet,

en fait

c’était une cloche »

- Menteur : la citation est apocryphe, c’est toi qui l’a générée !

Discriminateur, va ;)

Références et liens supplémentaires :

- Le Portrait d’Edmond de Belamy : 
https://fr.wikipedia.org/wiki/Portrait\_d’Edmond\_de\_Belamy

- La présentation des GANs : https://fr.wikipedia.org/wiki/Réseaux\_antagonistes\_génératifs

- Des présentations alternatives complémentaires des GANs :

 
 
 
 
- https://www.lebigdata.fr/gan-definition-tout-savoir

- https://www.lesechos.fr/tech-medias/intelligence-artificielle/les-gan-repoussent-les-limites-
de-lintelligence-artificielle-206875

- En savoir plus sur ces réseaux de calcul “profonds” https://pixees.fr/ce-quon-appelle-le-
deep-learning/

- À propos de création artistique et numérique https://pixees.fr/mais-que-peut-on-appeler-
creation-artistique

