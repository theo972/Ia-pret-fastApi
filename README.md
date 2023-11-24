# Ia-pret-fastApi

Loans

L'objectif est de déterminer le taux de personnes susceptibles de rembourser le prêt.
Pour se faire nous avons utilisé les données fournies par https://www.kaggle.com/datasets/itssuru/loan-data/data.

Tout d'abord il est important de notifier qu'on ne peut pas se fier totalement à notre jeux de
donnée car on ne sait pas comment, par qui et avec quelle attention elles ont été collectées.

Même si notre modèle est correctement entrainé et remonte une performance de prédiction de 99%,
l'inconnue sur la récupération des données ne nous permet pas d'utiliser notre modèle pour des
prédictions réelles car les données d'entrainements peuvent être fausses.

Une donnée sur le % de prêt précédement remboursé par un emprunteur aurait été une donnée très
utile pour entrainer notre modèle. Cependant cette donnée ne figure pas dans le jeux de donnée.

Colonnes:

* <u>not.fully.paid</u> (**OBJECTIF**, bool): Indique si le prêt a été remboursé ou non.

* <u>credit.policy</u> (~~Inutilisé~~, bool): Indique si le client répond aux critères de souscription de crédit

* <u>Purpose</u> (~~Inutilisé~~, string): Raison de l’emprunt, inutilisé car textuel.

* <u>Int.rate</u> (~~Inutilisé~~, number(]0;1[)): Taux d’intérêt du prêt en question. Inutilisé car la somme est déjà présente

* <u>Installment (number)</u> : La somme à payer par mois

* <u>log.annual.in (number)</u> : Le logarithme naturel du revenu annuel auto-déclaré de l'emprunteur. ????

* <u>Dti (number)</u> : Le ratio dette/revenu de l'emprunteur

* <u>Fico</u> : Le score FICO varie généralement de 300 à 850, avec des plages de catégories de crédit définies comme suit :
  - 300-579 : Mauvais crédit (Cat. 0)
  - 580-669 : Crédit équitable (Cat. 1)
  - 670-739 : Bon crédit (Cat. 2)
  - 740-799 : Très bon crédit (Cat. 3)
  - 800-850 : Excellent crédit (Cat. 4)

* <u>days.with.cr.line (~~Inutilisé~~, number)</u> : Nombre de jours depuis lesquels l'emprunteur dispose d’un crédit. Inutilisé car nous considérons que toutes les échéances sont terminées.

* <u>revol.bal</u> : Solde renouvellable de l'emprunteur, aka le montant impayé à la fin du cycle de facturation de la carte de crédit

* <u>revol.util</u> : Le taux d'utilisation de la ligne de crédit renouvelable de l'emprunteur AKA le montant de la ligne de crédit utilisée par rapport au total du crédit disponible.

* <u>inq.last.6mths</u> : Nombre de demandes de renseignements/plaintes de la part des créanciers de l'emprunteur au cours des six derniers mois.

* <u>delinq.2yrs</u> : Le nombre de fois où l'emprunteur a été en retard de paiement de plus de 30 jours au cours des deux dernières années.

* <u>pub.rec</u> : Le nombre de dossiers publics dérogatoires de l'emprunteur (dépôts de bilan, privilèges fiscaux ou jugements).
