# Mini-projet VRP-TSP

Ces programmes ont été réalisés dans le cadre du cours électif "Smart Decision" de l'Ecole Centrale de Lille par Mathis RIMBERT et Samy BENZAIM durant l'année scolaire 2023/2024. 

Une documentation détaillée du code est présente sous format PDF ainsi que sur ce fichier. Il est également possible de consulter le code source commenté. Pour toutes questions ou remarques, me contacter à  [ce lien](mailto:mathis.rimbert@centrale.centralelille.fr)

Repo GitHub associé à ce projet : [VRP-TSP](https://github.com/mrimbert/VRP-TSP)

## Sommaire

1. Problème du voyageur de commerce : TSP
2. Vehicule Routing Problem : VRP

## Problème du voyageur de commerce

Pour la réalisation de ce problème, deux algorithmes ont été développés, l'algorithme du recuit simulé ainsi que l'algorithme dit de la colonie de fourmis. Les codes sources détaillés sont accessibles dans les archives dédiées. Par manque de temps, seul une génération de ville aléatoire a été développée. Néanmoins, le code est prêt pour accueillir un fichier au format .CSV afin d'ajouter les coordonnées des villes souhaitées (le code est facilement adaptable pour cet usage). L'interface graphique permet de regénérer le nombre de villes souhaitées ainsi que la ville de départ. 

## Vehicule Routing Problem : VRP

Pour la réalisation de ce problème, l'algorithme génétique a été utilisé. Le code source est accessible pour visualiser l'implémentation de cette solution. A nouveau, seule une génération aléatoire des villes à été développée. Néanmoins, une implémentation à l'aide d'un fichier CSV de ville prédéfinie est facilement faisable (mais chronophage...)

## Remarque vis à vis de la présence d'un fichier exécutable

Un fichier exécutable Windows a été réalisé également pour lancer ce programme, il n'est néanmoins pas disponible sur le Repo GitHub du projet pour privilégier uniquement le code source qui est largement suffisant pour exécuter correctement le projet (le fichier requirements.txt contient l'intégralité des libraries utilisées par le programme).