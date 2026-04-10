# Ce dépot présente le projet de recherche "Détection d'objets par caméra neuromorphique"

## Ressources :

- Dépôt github du YOLO spiké : https://github.com/BICLab/SpikeYOLO

- Datasets prophesee : https://docs.prophesee.ai/stable/datasets.html (Un dataset réduit de Gen4 existe sous le nom de "Simplified mini-dataset")

## Description du projet :

Les caméras neuromorphiques réagissent aux changements de luminosité sur chaque pixel. Contrairement à une caméra classique, elle n'envoie pas image par image, mais des valeurs de changement de luminosité pixel par pixel, avec la timestamp de ce changement (à la microseconde près)

L'avantage de ce type de caméra est la vitesse. Elle peut envoyer 1 changement par micro-seconde, soit 1 000 000 de changements par seconde. Cela permet donc d'obtenir des informations beaucoup plus rapidement qu'une caméra classique qui envoie du 60 images par seconde.

En plus de cet avantage de rapidité, ce type de caméra consomme moins d'énergie qu'une caméra classique. Cela combiné avec un réseau de neurones spiké permet de diminuer considérablement la consommation énergétique d'un tel système.

L'une des applications de cette caméra pourrait être la voiture autonome. Un temps de réaction plus rapide pour réagir à son environnement pourrait permettre à la voiture d'éviter des accidents. C'est donc dans ce cadre que la reconnaissance d'objets de la route sur des données provenant de ce type de caméra intervient.

## Réseau de neurone normal VS spiké
- Dans un réseau de neurones normal, chaque neurone a une valeur qui est pondérée pour être prise en compte par les neurones de la couche suivante.

- Pour un réseau de neurones spiké, c'est un peu différent. Les valeurs envoyées d'un neurone vers la couche suivante ne peuvent être que 1 ou 0. Les neurones vont accumuler un potentiel. Lorsque ce potentiel atteint une valeur critique, un "spike" va se déclencher, envoyer 1 à la couche suivante, et la valeur du neurone va retourner à 0. Dans ce type de réseaux de neurones, les valeurs à déterminer sont les valeurs critiques pour chaque neurone.

- Le grand avantage d'un réseau de neurones spiké est sa très faible consommation en énergie comparé à un réseau de neurone classique.

## Prise en main du gihub SpikeYOLO

- Dans SpikeYOLO_for_Gen1/ultralytics/cfg/datasets, mettre le fichier gen4.yaml

- Dans SpikeYOLO_for_GEN1/train.py, remplacer gen1.yaml par gen4.yaml dans les paramètres de la fonction train.

- Changer le nombre de classes à détecter (7 classes pour gen4) dans le fichier SpikeYOLO_for_GEN1/ultralytics/cfg/models/v8/snn_yolov8.yaml. C'est dans ce fichier que vous pouvez retrouver la structure du réseau de neurones.

- Lancer l'entrainement : exécuter le fichier SpikeYOLO_for_GEN1/train.py

- Tester le modèle : préparer le fichier SpikeYOLO_for_GEN1/test.py, mettre le bon chemin vers les poids de votre modèle, changer le "gen1.yaml" en "gen4.yaml", et exécuter le fichier

- Valider le modèle : SpikeYOLO_for_GEN1/test.py, remplacer "gen1.yaml" par "gen4.yaml", mettre le bon chemin vers les poids du réseau de neurones, et exécuter le fichier.

- Hyperparamètres : Tous les hyperparamètres sont disponibles dans le fichier SpikeYOLO_for_Gen1\ultralytics\cfg\default.yaml


## Prétraitement des données :

- Pour atteindre une rapidité de réponse, il faut aggréger les données sur une fenêtre de temps faible. C'est l'objectif du fichier "Traitement fichier.py". Il permet de pré-traiter les données raw pour les aggréger en fenêtres de temps plus ou moins longues. On peut ainsi créer plusieurs datasets pour tester le changement qu'un pré-traitement plus ou moins long peut induire sur le spikeYOLO.