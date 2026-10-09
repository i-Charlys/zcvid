# Spécification & Architecture : Profil de Transport Vidéo Différentiel Ternaire sur UDP (`zcrudp`)

## Executive Summary & Context

`zcrudp` est un protocole de transport réseau sur UDP écrit en C11, conçu spécifiquement pour des applications ultra-basse latence (fréquences de rafraîchissement de **120 Hz à 240 Hz**), hautement déterministes et garanties **sans allocation dynamique (`malloc`-free)**.

Dans le cadre du transport vidéo à ultra-basse latence, les représentations vidéo conventionnelles (telles que les trames complètes Uncompressed RGB/YUV ou les flux inter-prédits H.264/HEVC) présentent des inconvénients majeurs :
1. **Bande passante critique** pour les flux bruts non compressés à 240 Hz.
2. **Latence de codage/décodage non déterministe** et coût CPU/GPU élevé pour les codecs vidéo hybrides standard.
3. **Sensibilité extrême aux pertes de paquets** : un paquet manquant dans une trame de référence (I-frame ou P-frame) provoque une propagation d'erreur catastrophique.

Ce document étudie la faisabilité et spécifie l'architecture d'un **profil vidéo différentiel ternaire** $\{-1, 0, +1\}$ exploité sur les **canaux non fiables / séquencés** de `zcrudp`.

---

## 1. Modèle de Représentation & État de l'Art (DVS & ADM)

### 1.1 Modulation Delta Adaptive (ADM) et Capteurs Neuromorphiques (DVS)

#### Modulation Delta (DM) & ADM
La Modulation Delta (DM) est une technique de conversion analogique-numérique différentielle où le signal est représenté non pas par sa valeur absolue, mais par une suite de variations élémentaires (ou pas quantifiés) $+\Delta$ ou $-\Delta$. Dans sa forme ternaire $\{-1, 0, +1\}$, un trit $0$ est introduit pour représenter l'absence de variation significative (zone sous le seuil de détection ou état stationnaire).

$$\Delta S(x, y, t) = \begin{cases}
+1 & \text{si } I(x, y, t) - I(x, y, t-\delta t) > \theta_{up} \\
-1 & \text{si } I(x, y, t) - I(x, y, t-\delta t) < -\theta_{dn} \\
0 & \text{sinon}
\end{cases}$$

#### Capteurs Evénementiels / Neuromorphiques (DVS - Dynamic Vision Sensors)
Les capteurs événementiels neuromorphiques (ex. Prophesee Metavision, Sony IMX636) s'affranchissent du concept de trame globale (frame-rate). Chaque pixel réagit de manière autonome à la variation de log-luminance $\ln(I)$. Dès que la variation dépasse un seuil $\theta$, un événement assynchrone $(x, y, t, p)$ avec polarité $p \in \{-1, +1\}$ est émis.

À haute fréquence (120 Hz – 240 Hz), une caméra conventionnelle à obturateur global (Global Shutter) capturant des changements temporels fins peut être discrétisée dans le domaine ternaire $\{-1, 0, +1\}$ à chaque *tick* temporel ($\Delta t = 4.16\text{ ms}$ à $8.33\text{ ms}$).

### 1.2 Propriétés de la vidéo ternaire à 120 Hz – 240 Hz
* **Sparsité spatio-temporelle extrême** : Dans la majorité des scènes réelles (robotique, vision industrielle, gaming cloud, télépilotage), entre **80% et 98% des pixels** ne changent pas d'état entre deux ticks consécutifs à 240 Hz (valeur $0$).
* **Bornage strict** : Chaque élément de variation vaut exactement un trit parmi trois états possibles $\{-1, 0, +1\}$.

---

## 2. Compaction Mathématique et Encodage Ternaire Sans Perte

Pour maximiser l'efficacité de l'Unité de Charge Utile Atomique **TFV 8/8/16** (`uint8_t type`, `uint8_t flags`, `uint16_t value`), les trits $\{-1, 0, +1\}$ doivent être compactés dans les champs d'octets sans gaspillage d'espace bit par bit.

### 2.1 Représentation Interne des Trits
Pour éliminer le traitement des nombres signés lors des opérations logiques/arithmétiques bit à bit, nous mappons les trits signés $\{-1, 0, +1\}$ sur des valeurs entières non signées $\{0, 1, 2\}$ :

$$t_i \in \{-1, 0, +1\} \implies v_i = t_i + 1 \in \{0, 1, 2\}$$

* Trit $-1 \implies 0_3$
* Trit $0 \implies 1_3$
* Trit $+1 \implies 2_3$

### 2.2 Compaction de 5 Trits dans 8 Bits (`uint8_t`)

Un ensemble de 5 trits possède $3^5 = 243$ combinaisons d'états uniques.
Puisque $243 \le 2^8 = 256$, **5 trits s'insèrent exactement dans un seul octet `uint8_t`**.

#### Équation de Colisage (Packing)
Pour 5 trits $v_0, v_1, v_2, v_3, v_4 \in \{0, 1, 2\}$ :

$$B = v_0 + 3 \cdot v_1 + 9 \cdot v_2 + 27 \cdot v_3 + 81 \cdot v_4$$

$$\text{avec } B \in [0, 242] \subset [0, 255]$$

#### Équation de Décolisage (Unpacking)
Pour extraire les trits à partir de l'octet $B$ :

$$v_0 = B \bmod 3$$
$$v_1 = (B / 3) \bmod 3$$
$$v_2 = (B / 9) \bmod 3$$
$$v_3 = (B / 27) \bmod 3$$
$$v_4 = (B / 81) \bmod 3$$

### 2.3 Compaction de 10 Trits dans 16 Bits (`uint16_t`)

Le champ `value` de la structure TFV (8/8/16) fait exactement **16 bits** (`uint16_t`).
Un ensemble de 10 trits possède $3^{10} = 59\,049$ combinaisons d'états uniques.
Puisque $59\,049 \le 2^{16} = 65\,536$, **10 trits s'insèrent parfaitement dans un mot de 16 bits**.

#### Équation de Colisage (Packing)
Pour 10 trits $v_0, \dots, v_9 \in \{0, 1, 2\}$ :

$$W = \sum_{i=0}^{9} v_i \cdot 3^i$$

$$\text{avec } W \in [0, 59048] \subset [0, 65535]$$

#### Équation de Décolisage (Unpacking)
Pour extraire les 10 trits à partir du mot $W$ :

$$v_i = \left( \frac{W}{3^i} \right) \bmod 3 \quad \forall i \in \{0, \dots, 9\}$$

### 2.4 Efficacité Spatiale et Comparaison

| Format | Capacité théorique | Valeurs max combinatoires | Espace disponible | Efficacité Spatiale | Alignement Mémoire |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5 trits / 8 bits** | $3^5 = 243$ | $243$ | $256$ ($2^8$) | $94.92\%$ | Octet (`uint8_t`) |
| **10 trits / 16 bits** | $3^{10} = 59\,049$ | $59\,049$ | $65\,536$ ($2^{16}$) | $90.10\%$ | Mot 16-bit (`uint16_t` TFV value) |
| **2-bit naïf (20 trits/32b)** | $2^{20} = 1\,048\,576$ | $3^{10} \times 3^{10}$ | - | $58.50\%$ (gaspillage de 25% de bits) | Non optimal |

L'encodage en base 3 permet d'économiser jusqu'à **25% de bande passante** par rapport à l'encodage naïf 2-bits par trit (qui ne permettrait de mettre que 8 trits dans 16 bits au lieu de 10).

---

## 3. Analyse de la Dérive (Drift) et Résilience sur Canal Non Fiable

Dans un canal UDP non fiable, la perte d'un paquet contenant des variations de deltas $\{-1, 0, +1\}$ entraîne une désynchronisation entre l'état du canevas de l'émetteur et celui du récepteur. Si la reconstruction est une simple intégration pure ($I(t) = I(t-1) + \Delta I(t)$), l'erreur s'accumule indéfiniment (**Drift**).

Deux approches de résilience sont modélisées et comparées.

### 3.1 Approche A : Intra-Refresh / Tranches d'Ancres Absolues sur Canal Fiable

#### Principe
* La vidéo est découpée en tranches horizontales ou macro-lignes.
* À chaque tick, pendant que le canal non fiable transporte les deltas ternaires $\{-1, 0, +1\}$ de l'image entière, une **tranche d'ancre absolue** (valeurs absolues de pixels Y/Luma 8 bits) est envoyée sur le **canal fiable/ordonné** de `zcrudp`.
* Sur un cycle de $N$ frames (ex. $N=12$ ticks à 240 Hz = 50 ms), 100% des lignes du canevas récepteur sont réinitialisées avec certitude.

#### Formulation Mathématique
Soit $Y(x, y, t)$ la valeur absolue reconstruite du pixel au point $(x, y)$ et au temps $t$.
Si le pixel $(x, y)$ appartient à la tranche rafraîchie $S_t$ transmise de façon fiable à $t$ :

$$Y(x, y, t) = Y_{absolute}(x, y, t)$$

Pour tous les autres pixels $(x,y) \notin S_t$ reçus via le canal non fiable :

$$Y(x, y, t) = \text{clamp}_{0..255} \left( Y(x, y, t-1) + \delta \cdot \Delta S(x, y, t) \right)$$

#### Avantages et Inconvénients
* **Avantages** : Reconstitution stricte et exacte sans perte de dynamique ni flou spatial. Élimination totale du drift en $N$ ticks.
* **Inconvénients** : Consomme une portion de la bande passante du canal fiable (`ACK`/retransmission).

### 3.2 Approche B : Intégrateur à Fuite (Leaky Integrator) sur Canal Non Fiable

#### Principe
Pour fonctionner **strictement sur canal non fiable** sans exiger le moindre ACK ni réémission de données, le récepteur applique une atténuation exponentielle (ou fuite) constante sur le canevas vidéo récepteur à chaque tick.

En l'absence de nouveau delta, la valeur des pixels dérive naturellement vers une valeur de gris neutre / fond moyen $Y_{neutral} = 128$. Les erreurs causées par la perte de paquets sont amorties de façon exponentielle et disparaissent visuellement au bout de quelques constantes de temps $\tau$.

#### Formulation Mathématique
Soit $\gamma \in (0, 1)$ le facteur de fuite (ex. $\gamma = 0.95$ ou $\gamma = 0.98$), et $Y_{neutral} = 128$ :

$$Y(x, y, t) = \text{clamp}_{0..255} \left( Y_{neutral} + \gamma \cdot \big(Y(x, y, t-1) - Y_{neutral}\big) + \delta \cdot \Delta S(x, y, t) \right)$$

Si un paquet est perdu au tick $t$, le récepteur effectue la fuite sans ajouter de delta ($\Delta S = 0$) :

$$Y(x, y, t) = Y_{neutral} + \gamma \cdot \big(Y(x, y, t-1) - Y_{neutral}\big)$$

Soit $E_0$ l'erreur d'amplitude introduite par un paquet perdu au temps $t_0$. Au bout de $k$ ticks, l'erreur restante est :

$$E(k) = E_0 \cdot \gamma^k$$

Pour $\gamma = 0.90$ à 240 Hz ($t_{tick} = 4.16 \text{ ms}$) :
* Après 10 ticks ($41.6 \text{ ms}$), $E(10) = E_0 \cdot (0.90)^{10} \approx 0.348 \cdot E_0$ (65% d'erreur résorbée).
* Après 20 ticks ($83.3 \text{ ms}$), $E(20) \approx 0.121 \cdot E_0$ (88% d'erreur résorbée).

#### Avantages et Inconvénients
* **Avantages** : Zero dépendance au canal fiable, zéro retransmission, complexité $O(1)$ uniforme.
* **Inconvénients** : Effet de rémanence / "fading" si $\gamma$ est trop faible; résidu visuel temporaire si $\gamma$ est très élevé.

### 3.3 Synthèse Comparative

| Critère | Intra-Refresh (Canal Fiable) | Leaky Integrator (Canal Non Fiable Pure) |
| :--- | :--- | :--- |
| **Garantie d'absence de drift** | Absolue (après 1 cycle complet $N$) | Asymptotique (atténuation exponentielle) |
| **Charge Réseau Fiable** | Nécessite un canal ACK / fiable | **Zero** charge sur canal fiable |
| **Déterminisme Temporel** | Sensible au Jitter si retransmissions ACK | **Déterminisme strict 100%** |
| **Coût Calculateur Récepteur** | Copie directe pour tranche d'ancre | 1 multiplication / fondu par pixel |
| **Cas d'usage optimal** | Vidéo de précision (télépilotage, chirurgie) | Vision industrielle ultra-rapide / Suivi de mouvement |

---

## 4. Format de Payload et Intégration Protocolaire `zcrudp`

### 4.1 Rappel de la Trame et Record TFV `zcrudp`

Chaque record dans le bundler intra-tick respecte la structure atomique 32 bits TFV 8/8/16 :

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Type (8)    |   Flags (8)   |          Value (16)           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### 4.2 Spécification des Types de Records du Profil Ternaire

Nous définissons deux types de records dédiés au profil vidéo ternaire (Track D - Ternary Video) :

1. **`ZCRUDP_TFV_TYPE_TERNARY_HEADER` (Type `0xD0`)** : En-tête de bloc / tranche vidéo.
   * `type` : `0xD0`
   * `flags` : Numéro de tranche ou d'index de bloc ($0 .. 255$).
   * `value` : `uint16_t` contenant les coordonnées de départ de la ligne / bloc ($Y$ offset ou $X, Y$).

2. **`ZCRUDP_TFV_TYPE_TERNARY_PACK10` (Type `0xD1`)** : Record de données 10 trits compactés.
   * `type` : `0xD1`
   * `flags` : Offset de séquence ou masque de validité spatiale dans la tranche.
   * `value` : `uint16_t` contenant les 10 trits compactés en base 3 ($0 .. 59048$).

3. **`ZCRUDP_TFV_TYPE_TERNARY_PACK5_PAIR` (Type `0xD2`)** : Record alternatif contenant deux paquets de 5 trits.
   * `type` : `0xD2`
   * `flags` : Offset de séquence.
   * `value` : `uint16_t` combinant deux octets $B_0, B_1$ ($B_0 | (B_1 \ll 8)$) contenant chacun 5 trits.

### 4.3 Structure Sans Allocation Dynamique (`malloc`-free)

En C11, les structures de données réceptrices et émettrices sont pré-allouées statiquement pour correspondre à la mémoire maximale du canevas et du MTU (ex. 1472 octets utile UDP).

```c
typedef struct {
    uint8_t  type;    /* ZCRUDP_TFV_TYPE_TERNARY_* */
    uint8_t  flags;   /* Sub-stream ID / Sequence offset */
    uint16_t value;   /* 10-trit base-3 encoded data or header metadata */
} zcrudp_tfv_record_t;
```

---

## 5. Calculs des Budgets de Bande Passante Théoriques

### 5.1 Paramètres du Calcul
* **Dimensions Résolution** :
  * SD Low-Res : $640 \times 360 = 230\,400$ pixels.
  * HD Ready : $1280 \times 720 = 921\,600$ pixels.
* **Fréquences de Rafraîchissement** : $120 \text{ Hz}$ ($\Delta t = 8.33 \text{ ms}$) et $240 \text{ Hz}$ ($\Delta t = 4.16 \text{ ms}$).
* **Activité de la Scène ($\alpha$)** : Pourcentage de pixels subissant une variation non nulle $\{-1, +1\}$ au tick courant.
  * Scène quasi-statique (surveillance, fond fixe) : $\alpha = 2\%$ ($0.02$).
  * Scène à mouvement moyen (robotique d'intérieur, télépilotage) : $\alpha = 5\%$ ($0.05$).
  * Scène hautement dynamique (mouvements caméra rapides, action) : $\alpha = 15\%$ ($0.15$).

### 5.2 Ratios de Compaction
* Avec **10 trits par `uint16_t` (record 32-bit TFV)** :
  * 1 record TFV (4 octets) transporte 10 trits (soit 10 pixels).
  * Coût moyen brut par pixel en mode dense : $4 \text{ bits/pixel}$ ($0.5 \text{ octet/pixel}$).
* Avec **Codage Sparsifié (Adresse de bloc + 10 trits)** :
  * Pour un bloc actif de 10 pixels consécutifs, on émet 1 record TFV.

### 5.3 Tableau des Budgets de Bande Passante (Dense vs Sparsifié)

#### Mode A : Dense (Transmission systématique de tous les pixels)
Chaque trame envoie tous les pixels compactés à 10 trits / 32 bits (4 octets pour 10 pixels = $0.4$ octet/pixel).

* **640x360 (230 400 pixels)** :
  * Taille trame = $230\,400 / 10 \times 4 \text{ octets} = 92\,160 \text{ octets}$.
  * À $120 \text{ Hz}$ : $92\,160 \times 120 \times 8 = \mathbf{88.47 \text{ Mbps}}$.
  * À $240 \text{ Hz}$ : $92\,160 \times 240 \times 8 = \mathbf{176.95 \text{ Mbps}}$.

* **1280x720 (921 600 pixels)** :
  * Taille trame = $921\,600 / 10 \times 4 \text{ octets} = 368\,640 \text{ octets}$.
  * À $120 \text{ Hz}$ : $368\,640 \times 120 \times 8 = \mathbf{353.89 \text{ Mbps}}$.
  * À $240 \text{ Hz}$ : $368\,640 \times 240 \times 8 = \mathbf{707.79 \text{ Mbps}}$.

#### Mode B : Sparsifié (Transmission basée sur l'Activité $\alpha$)
Seuls les blocs de 10 pixels contenant au moins un delta non nul sont transmis, précédés d'un header de bloc.

* **640x360 à 240 Hz** :
  * $\alpha = 2\%$ ($\sim 4608$ pixels actifs $\implies 461$ blocs) : **$\approx 3.54 \text{ Mbps}$**.
  * $\alpha = 5\%$ ($\sim 11520$ pixels actifs $\implies 1152$ blocs) : **$\approx 8.85 \text{ Mbps}$**.
  * $\alpha = 15\%$ ($\sim 34560$ pixels actifs $\implies 3456$ blocs) : **$\approx 26.54 \text{ Mbps}$**.

* **1280x720 à 240 Hz** :
  * $\alpha = 2\%$ : **$\approx 14.15 \text{ Mbps}$**.
  * $\alpha = 5\%$ : **$\approx 35.39 \text{ Mbps}$**.
  * $\alpha = 15\%$ : **$\approx 106.17 \text{ Mbps}$**.

Cette analyse démontre que l'association de l'encodage ternaire compacté et du canal non fiable UDP offre une adéquation parfaite pour le transport vidéo ultra-basse latence sur des liens Ethernet Gigabit ou Wi-Fi 6/7.

---

## 6. Guide d'Implémentation C11 (`include/protocol_ternary.h`)

La bibliothèque C11 doit respecter les contraintes suivantes :
1. Dépliage de boucle (loop unrolling) à la compilation.
2. Utilisation d'arithmétique entière pure (tables de puissances de 3 précalculées).
3. Utilisation de modulos et divisions optimisés.
4. Compatibilité vectorielle SIMD (Auto-vectorisation GCC/Clang via `-O3`).

---

## 7. Conclusion & Recommandations
1. **Compaction** : Utiliser l'encodage 10 trits dans 16 bits (`uint16_t`) comme format primaire de charge utile TFV pour atteindre $90.1\%$ d'efficacité spatiale théorique.
2. **Résilience** : Implémenter l'atténuation exponentielle (Leaky Integrator, $\gamma = 0.95$) sur le canal non fiable par défaut pour garantir la tolérance aux pertes de paquets sans retransmission.
3. **Architecture C11** : Conserver un design header-only/inline zero-allocation compatible avec les contraintes temps réel de `zcrudp`.
