import numpy as np

# --- Sample encoded client row ---
sample_row = {
    "Genre_Femme": 1,
    "Genre_Homme": 0,
    "Moyen d'obtention de salaire_Directement auprès de l'opérateur": 0,
    "Moyen d'obtention de salaire_Virement bancaire": 1,
    "Type du crédit demandé_Crédit de consommation": 1,
    "Type du crédit demandé_Crédit immobilier": 0,
    "Type du crédit demandé_Crédit- auto": 0,
    "Avez vous un crédit en cours?_Crédit pour la première fois": 1,
    "Avez vous un crédit en cours?_J'ai déjà un crédit bancaire et je veux un autre": 0,
    "Avez vous un crédit en cours?_L'ancien crédit est remboursé et je veux un autre": 0,
    "Type du crédit obtenu_Crédit au démarrage du projet": 0,
    "Type du crédit obtenu_Crédit de consommation": 1,
    "Type du crédit obtenu_Crédit immobilier": 0,
    "Type du crédit obtenu_Crédit- auto": 0,
    "Type du crédit obtenu_Micro-crédit": 0,
    "Type du crédit obtenu_crédit de consommation": 0,
    "Type du crédit obtenu_pas de crédit obtenu": 0,
    "Avez-vous dépassé la moitié de la période totale de remboursement_non": 0,
    "Avez-vous dépassé la moitié de la période totale de remboursement_oui": 1,
    "Avez-vous dépassé la moitié de la période totale de remboursement_pas de crédit obtenu": 0,
    "Avez-vous des cessions sur salaire?_non": 1,
    "Avez-vous des cessions sur salaire?_oui": 0,
    "Income Category_Good Income": 1
}

# --- Logistic Regression Scoring Function ---
def compute_logistic_score(row):
    return -1.2091 \
        + (0.0872 * row.get("Genre_Femme", 0)) \
        + (0.1478 * row.get("Genre_Homme", 0)) \
        + (-1.9765 * row.get("Moyen d'obtention de salaire_Directement auprès de l'opérateur", 0)) \
        + (2.2474 * row.get("Moyen d'obtention de salaire_Virement bancaire", 0)) \
        + (0.0418 * row.get("Type du crédit demandé_Crédit de consommation", 0)) \
        + (0.0194 * row.get("Type du crédit demandé_Crédit immobilier", 0)) \
        + (0.0259 * row.get("Type du crédit demandé_Crédit- auto", 0)) \
        + (0.6370 * row.get("Avez vous un crédit en cours?_Crédit pour la première fois", 0)) \
        + (-0.3333 * row.get("Avez vous un crédit en cours?_J'ai déjà un crédit bancaire et je veux un autre", 0)) \
        + (-0.0716 * row.get("Avez vous un crédit en cours?_L'ancien crédit est remboursé et je veux un autre", 0)) \
        + (0.1188 * row.get("Type du crédit obtenu_Crédit au démarrage du projet", 0)) \
        + (0.0709 * row.get("Type du crédit obtenu_Crédit de consommation", 0)) \
        + (-0.0244 * row.get("Type du crédit obtenu_Crédit immobilier", 0)) \
        + (-0.0036 * row.get("Type du crédit obtenu_Crédit- auto", 0)) \
        + (0.0293 * row.get("Type du crédit obtenu_Micro-crédit", 0)) \
        + (0.0064 * row.get("Type du crédit obtenu_crédit de consommation", 0)) \
        + (0.6924 * row.get("Type du crédit obtenu_pas de crédit obtenu", 0)) \
        + (-3.1489 * row.get("Avez-vous dépassé la moitié de la période totale de remboursement_non", 0)) \
        + (2.6378 * row.get("Avez-vous dépassé la moitié de la période totale de remboursement_oui", 0)) \
        + (0.6924 * row.get("Avez-vous dépassé la moitié de la période totale de remboursement_pas de crédit obtenu", 0)) \
        + (0.2040 * row.get("Avez-vous des cessions sur salaire?_non", 0)) \
        + (0.0718 * row.get("Avez-vous des cessions sur salaire?_oui", 0)) \
        + (1.6208 * row.get("Income Category_Good Income", 0))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Evaluate ---
score = compute_logistic_score(sample_row)
probability = sigmoid(score)

print("\n--- Credit Scoring Result ---")
print("Logistic Score:", round(score, 2))
print("Approval Probability:", f"{probability * 100:.2f}%")
