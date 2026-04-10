# Ce dépot présente le projet de recherche "Détection d'objets par caméra neuromorphique"

## Ressources :

- Dépôt github du YOLO spiké : https://github.com/BICLab/SpikeYOLO

- Datasets prophesee : https://docs.prophesee.ai/stable/datasets.html (Un dataset réduit de Gen4 existe sous le nom de "Simplified mini-dataset")

### Bibliographie :

2025
1.	Wang et al.
Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset
Conférence : CVPR 2025
Lien : https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Object_Detection_using_Event_Camera_A_MoE_Heat_Conduction_based_CVPR_2025_paper.html
2.	Li et al.
Brain-Inspired Spiking Neural Networks for Energy-Efficient Object Detection
Conférence : CVPR 2025
Lien : https://openaccess.thecvf.com/content/CVPR2025/html/Li_Brain-Inspired_Spiking_Neural_Networks_for_Energy-Efficient_Object_Detection_CVPR_2025_paper.html
3.	Auteurs non précisés (Prophesee/Metavision)
Object Detection Method with Spiking Neural Network based on DT-LIF Neuron and SSD
Source : Prophesee Technical Report
Lien : https://www.prophesee.ai/2025/04/09/object-detection-method-with-spiking-neural-network-based-on-dt-lif-neuron-and-ssd/
4.	Ahmed et al.
Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention
Conférence : CVPR 2025
Lien : https://arxiv.org/abs/2403.10173
5.	Wu et al.
Spiking Transformer-CNN for Event-based Object Detection
Conférence : OpenReview/NeurIPS 2025
Lien : https://openreview.net/forum?id=zweyouirw7
2024
6.	Wu et al.
LEOD: Label-Efficient Object Detection for Event Cameras
Conférence : CVPR 2024
Lien : https://openaccess.thecvf.com/content/CVPR2024/html/Wu_LEOD_Label-Efficient_Object_Detection_for_Event_Cameras_CVPR_2024_paper.html
7.	Auteurs non précisés
Event Meta Formers for Event-based Real-time Traffic Object Detection
Source : arXiv
Lien : https://arxiv.org/abs/2504.04124
8.	Auteurs non précisés
Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection (SpikeYOLO)
Source : arXiv
Lien : https://arxiv.org/abs/2407.20708
2023
9.	Ge et al.
EMS-YOLO: A Directly Trained Spiking Neural Network for Object Detection
Source : arXiv
Lien : https://arxiv.org/abs/2307.11411
Cordone, L.
Performance of spiking neural networks on event data for embedded automotive applications (Chapter 5: No-reset object detection with SNN)
Source : Thèse PhD, code GitHub
Lien : https://github.com/loiccordone/no-reset-object-detection-with-snn


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