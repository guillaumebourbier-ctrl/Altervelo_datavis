SAE DATA : Mobilité Urbaine 

# **SAE DATA Mobilité Urbaine, Algorithmes & Éthique de l'IA** 

1 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

**Objectif :** Transformer des données brutes de vélos en libre-service en outils d'aide à la décision éthiques et performants. 

## **Villes suggérées :** 

**Paris** (Volume massif) 

**Saint-Pierre** (Topographie & Climat tropical) 

**Oslo** (Historique complet & saisonnalité marquée) 

2 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **Référentiel de Compétences - DATA** 

## **Traiter des données massives à l’aide des techniques d’intelligence artificielle** 

**Ingénierie & Qualité** : Appliquer les méthodologies adaptées de stockage, nettoyage et transformation. 

**Aide à la Décision** : Extraire des informations stratégiques 

**Communication** : Présenter les informations de manière adaptée aux publics visés (directions, clients, collaborateurs). 

**Éthique & Droit** : Intégrer les problématiques de sûreté, de respect de la vie privée et du droit d’auteur. 

**IA Responsable** : Mesurer et réduire l'impact environnemental des modèles d'IA. **Expertise** : Assurer une veille scientifique et technologique permanente pour ESIROI 3A - 2026 

3 

SAE DATA : Mobilité Urbaine 

## **Choix des "Packages"** 

. Chaque groupe, binôme, se voit attribuer un **Package thématique** 

## **Un package est constitué de 3 piliers :** 

1. **La Problématique Métier** : Un défi concret (logistique, météo, vol, etc.) à 

   - résoudre pour l'exploitant. 

2. **L'Algorithme Cible** : Un modèle spécifique à approfondir (théorie + pratique). 

3. **L'Enjeu Éthique IA** : Une réflexion critique sur l'usage de l'IA dans ce contexte précis. 

4 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

**Package 1 : Logistique & Disponibilité (G1 & G2) Problématique : Éviter la rupture de service Objectif** : Prédire si une station aura moins de 3 vélos à min. **Algorithmes** : **Random Forest** (G1) vs **XGBoost** (G2). **Éthique IA : La Boucle de Rétroaction** Si l'IA prédit qu'une station sera vide et qu'on y remet des vélos, on renforce artificiellement son usage. Comment évaluer un modèle qui modifie le comportement du système en temps réel ? 

5 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **Package 2 : Météo-Résilience (G3)** 

**Problématique : Impact des aléas climatiques Objectif** : Quantifier l'impact de la pluie, du vent et des températures sur la demande globale. 

: . **Algorithme Régression Linéaire Multiple Éthique IA : IA Verte vs IA Énergivore** Le vélo est une énergie propre, mais l'entraînement de modèles IA consomme des ressources. Comment justifier le coût carbone de l'IA par rapport au gain d'efficacité du service ? 

6 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **Package 3 : Cycles et Flux Temporels (G4 & G5)** 

**Problématique : Le pouls de la ville** 

**Objectif** : Identifier les rythmes (pendulaires, loisirs, événements) et prédire les volumes quotidiens/annuels. 

: . **Algorithme Séries Temporelles (ARIMA / SARIMA) Éthique IA : Interprétabilité & Boîte Noire** Si une IA décide de tarifer plus cher lors d'un pic prédit, comment l'expliquer à l'usager ? Peut-on sacrifier la transparence au profit de la précision ? 

7 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

**Package 4 : Topographie et Effort (G6) Problématique : L'influence du relief** 

**Objectif** : Analyser comment la pente influence la prise et le dépôt des vélos (effet ventouse des bas de collines). 

: . **Algorithme SVM (Support Vector Machine) Éthique IA : Biais et Exclusion Territoriale** Si l'IA "apprend" que les zones en pente sont moins rentables, elle risque de suggérer de supprimer des stations. Comment l'algorithme peut-il corriger les inégalités d'accès au lieu de les renforcer ? 

8 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

**Package 5 : Vol, Disparition et Risques (G7) Problématique : Sécuriser la flotte Objectif** : Identifier les trajets à haut risque de vol ou de non-retour (abandon hors station). 

: . **Algorithme Random Forest (Classification) Éthique IA : Profilage et Faux Positifs** Quelles conséquences si l'IA classe à tort un usager comme "voleur" ? Risque de discrimination géographique (stigmatisation de certains quartiers) et droit à l'erreur humaine. 

9 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **Package 6 : Clustering d'Usages (G8)** 

## **Problématique : Typologie des utilisateurs** 

**Objectif** : Segmenter les usagers sans données nominatives (Vélotafeurs, touristes, noctambules). 

: . **Algorithme K-Means Clustering** 

**Éthique IA : Désanonymisation et Vie Privée** Protection des données : même "anonymisée", une trace GPS peut révéler l'identité (domicile/travail). Où s'arrête l'analyse et où commence la surveillance ? 

10 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **Liste des Livrables** 

## **1. Présentation: Enjeux Éthiques & IA** 

Chaque groupe présentera, en amont des travaux techniques, une analyse de **5 minutes** sur le sujet éthique lié à son package. 

**Contenu :** Analyse du risque (biais, vie privée, responsabilité), état de l'art législatif (RGPD, IA Act) et propositions de "mitigation" (comment limiter le risque par la technique). 

11 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

**2. Analyse Exploratoire & Fouille de Données (EDA)** Format: Jupyter Notebook + rapport des analyses et conclusions. Avant toute modélisation, compréhension du jeu de données : **Analyse Univariée & Distributions** : Identification des lois de distribution des variables clés (durée de trajet, fréquentation par station). Traitement des données aberrantes (outliers). **Tests Statistiques** : Mise en œuvre de tests pour valider des hypothèses. **Réduction de Dimension (ACP)** : Analyse en Composantes Principales pour identifier les corrélations entre variables et isoler les facteurs expliquant la plus grande variance. **Clustering exploratoire** : Utilisation de méthodes non supervisées pour segmenter les stations ou les comportements d'usage avant la phase de ESIROI 3A - 2026prédiction. 

12 

SAE DATA : Mobilité Urbaine 

## **3. Architecture des Données** 

Format: rapport technique. 

**Schéma Entité-Relation (ERD)** : Modélisation conceptuelle montrant les entités (Stations, Trajets, Météo, Vélos) et leurs cardinalités. 

**Schéma Relationnel (Modèle Conceptuel de Données) :** Fournir le diagramme des tables (Stations, Trajets, Météo, vélos) avec les clés primaires/étrangères. 

13 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **4. Présentation 2: Fonctionnement de l'Algorithme** 

**5 minutes** : Explications du fonctionnement de l'outil imposé dans votre package (ex: Comment XGBoost itère sur les erreurs ? Comment ARIMA gère la saisonnalité ?). L'objectif est de faire comprendre le concept mathématique à vos camarades. 

14 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **5. Visualisation : Interface Utilisateur (UI)** 

**Dashboard Décisionnel** : Création d'une interface permettant de visualiser l'état du réseau. 

**Indicateurs Temps Réel** : Affichage des prédictions générées par vos algorithmes (ex: scores de disponibilité, alertes "stock critique"). 

15 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **6. Rapport Technique & Méthodologie** 

**Pipeline de traitement** : Description des étapes de nettoyage et de préparation des données. 

**Validation du modèle** : 

Calcul des performances, selon sujet (MAE, RMSE, F1-Score, Matrice de confusion). **Méthode de validation** : Utilisation de la _Cross-Validation_ (ou _Time-Series Split_ ) pour garantir la robustesse. 

**Prévention du surapprentissage (Overfitting)** : Justification des techniques employées pour assurer que le modèle reste reproductible sur de nouvelles données. 

16 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **7. Soutenance Finale** 

- Chaque groupe dispose de **10 minutes** de présentation suivies de **5 minutes** de questions/réponses. 

## **Structure attendue de la soutenance :** 

1. **Synthèse de la problématique :** Quel problème tentez-vous de résoudre ? 

2. **Méthodologie de Fouille :** Qu'avez-vous découvertes lors de l'EDA ? 

3. **Choix Techniques :** Justification de l'architecture de la base de données et de la configuration de l'algorithme. 

4. **Démonstration de l'Outil :** Présentation de votre interface utilisateur et des 

   - indicateurs d'aide à la décision. 

5. **Résultats & Validation :** Performance réelle du modèle et limites rencontrées. 

ESIROI 3A - 20266. **Conclusion Éthique :** Synthèse critique de l'usage de l'IA pour ce cas précis. 

17 

SAE DATA : Mobilité Urbaine 

## **Points d'Attention (1/2)** 

## **1. Disponibilité des Données** 

**N'attendez pas** : Testez vos flux d'API (temps réel) et téléchargez vos fichiers historiques dès que possible. 

**Vérification** : Réalisez un premier échantillonnage (extraits de données) pour confirmer que votre ville dispose des colonnes nécessaires à votre package (ex: géolocalisation, météo, ID vélos). 

L'EDA (Analyse Exploratoire) est votre boussole. Si les données sont trop bruitées ou incomplètes sur une ville, **changez de ville** avant la fin de la semaine 1. 

18 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **Points d'Attention (2/2)** 

## **2. Stratégie de Stockage : Accumuler pour Prédire** 

Pour les prédictions en temps réel, lancez un script de récupération (script de "scraping" ou stockage en DB) au plus vite afin de constituer votre propre historique d'entraînement. 

## **3. Anticipation des Dépendances** 

Vérifiez la compatibilité entre vos sources (ex: la station de météo est-elle assez proche des stations de vélos ?). 

Anticipez les pannes d'API ou les quotas de requêtes limités. 

19 

ESIROI 3A - 2026 

SAE DATA : Mobilité Urbaine 

## **Rétro-planning (4 Semaines)** 

**Semaine 1 : Immersion & Éthique.** Choix de la ville, récupération des données (API/CSV), réflexion éthique. 

**Présentation Ethique 15/04** 

**Semaine 2 : Exploration & Architecture.** Analyse exploratoire (EDA). Conception du schéma ERD. Recherches sur algorithme. 

**Présentation Algorithme 23/04** 

**Semaine 3 : Modélisation & UI.** Entraînement des modèles, validation croisée, développement du Dashboard. 

**Semaine 4 : Finalisation & Soutenance.** Rédaction du rapport technique 

**Soutenance Finale 06/05** 

ESIROI 3A - 2026 

20 

