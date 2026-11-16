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

---

## 8. Addendum : passer sous $\log_2 3$ bits par trit

### 8.1 Ce que la borne de la section 2 dit vraiment

Les $90{,}1\,\%$ d'efficacité de la section 2.4 comparent $1{,}6$ bit par trit à $\log_2 3 \approx 1{,}585$ bit. Cette borne n'est la limite que si les trois valeurs sont équiprobables et indépendantes. Ce n'est pas le cas d'une vidéo :

* dans un bloc transmis, la majorité des trits valent encore $0$ ;
* les trits non nuls se regroupent le long des contours en mouvement, et leurs voisins portent le plus souvent le même signe.

La limite de Shannon est l'entropie conditionnelle $H(T \mid \text{voisinage})$, pas $\log_2 3$. Rien n'est violé : le format à jetons laissait simplement cette marge inutilisée.

### 8.2 Codeur retenu

`include/ternary_entropy.h` implémente un codeur d'intervalles binaire adaptatif (plage 32 bits, probabilités 12 bits, adaptation en $1/32$).

| Élément | Codage |
| :--- | :--- |
| Blocs nuls entre deux blocs codés | longueur en Exp-Golomb adaptatif, contexte = longueur précédente |
| Trit nul ou non | 1 décision, 24 contextes : 2 voisins de gauche, 3 voisins de la ligne du dessus |
| Signe | 1 décision, 9 contextes : signe à gauche et signe au-dessus |
| Ancres (`INTRA_EC`) | prédiction gauche puis prédicteur médian JPEG-LS, erreur codée sans perte |

**Contrainte de résilience.** Le codeur et toutes les probabilités repartent de zéro à chaque paquet, et les contextes ne lisent que des trits du même paquet. Deux sources de contexte plus efficaces ont été écartées pour cette raison : le trit de la trame précédente et le pas ADM courant. Toutes deux dépendent de l'état du récepteur ; après une perte, le décodeur arithmétique se désynchroniserait et le reste du paquet serait décodé en bruit. Le contrat de la section 3 est donc conservé : un paquet perdu ne coûte que sa propre étendue (vérifié par `test_entropy_packet_loss_is_isolated`).

### 8.3 Mesures

Image reconstruite identique au bit près dans les deux modes (`test_entropy_matches_tokens`). Scène de test : texture lissée en panoramique, objet mobile, HUD fixe.

| Configuration | Deltas, jetons | Deltas, codés | Bits par trit codé | Ancres |
| :--- | ---: | ---: | ---: | ---: |
| GRAY8, ADM | 96 488 o | 53 759 o | 0,92 | ÷ 2,70 |
| NV12, ADM, chroma 2x | 123 188 o | 66 960 o | 0,91 | ÷ 3,19 |
| RGBA, pas fixe | 324 616 o | 135 423 o | 0,69 | ÷ 2,72 |
| NV12, MTU 300 octets | 62 216 o | 36 430 o | 1,02 | ÷ 2,86 |
| GRAY8, largeur 8200 (sans ligne du dessus) | 79 522 o | 53 889 o | 1,12 | ÷ 2,32 |

Ablation sur du 720p NV12 dense : sans le contexte de la ligne du dessus, le flux delta grossit d'environ $7\,\%$. Un estimateur de probabilité à deux vitesses n'a rien apporté par rapport à l'estimateur simple ; il a été retiré.

**Sur de vraies photographies.** `make bench-image IMG=photo.pgm` rejoue une fenêtre 1280x720 sur une image réelle. Cinq photographies 4K ramenées à 1920x1200, panoramique de 1 px par trame, bruit de $\pm 1$, seuil 3 :

| Flux | Jetons ou brut | Codé | Gain |
| :--- | ---: | ---: | ---: |
| Image clé | 8,07 bpp | 2,44 à 3,40 bpp | ÷ 2,4 à 3,3 |
| Ancres glissantes | 0,336 bpp | 0,102 à 0,146 bpp | ÷ 2,3 à 3,3 |
| Deltas | 1,48 à 1,52 bpp | 0,84 à 1,06 bpp | ÷ 1,4 à 1,8 |

Avec un bruit de $\pm 3$ et sans mouvement, les gains tombent à ÷ 1,5 sur les deltas et ÷ 1,9 à 2,2 sur les ancres : le bruit ne se compresse pas. Les scènes générées surestimaient donc le gain d'environ un quart ; la scène du benchmark 1080p (rectangle uni, 22,3 Mbit/s vers 0,57 Mbit/s) reste un cas extrême qui ne représente pas un contenu réel.

### 8.4 Coût et plafond CPU

**Chemin jetons.** Avec AVX2, l'encodeur et le décodeur traitent un bloc de 10 échantillons par étape : quantification, référence en boucle fermée et pas ADM d'un côté ; dépaquetage du jeton (une multiplication haute par $1/3$, $1/9$, $1/27$, $1/81$, exacte sous 243) et application de l'autre. `test_avx2_block_matches_scalar` compare 200 000 blocs aléatoires au code scalaire, et les deux compilations produisent les mêmes octets sur toutes les scènes.

| Un cœur, luma 1280x720, tous les blocs actifs (photographie) | Encodage | Décodage |
| :--- | ---: | ---: |
| Jetons, scalaire | 4,8 ms | 2,4 ms |
| Jetons, AVX2 | 1,2 ms | 0,5 ms |
| Codage entropique, AVX2 | 7,6 à 9,6 ms | 6,7 à 8,5 ms |

**Codeur entropique.** Son coût est proportionnel aux blocs codés : 55 à 70 ns par bloc et par côté sur un cœur, soit environ 4 ns par décision binaire, l'ordre de grandeur d'un codeur de type CABAC. Ce qui a été essayé :

| Essai | Effet mesuré | Décision |
| :--- | :--- | :--- |
| État du codeur en registres pendant un bloc | décodage $-10\,\%$ | gardé |
| Contextes de la ligne du dessus calculés une fois par bloc (3 chargements de 16 octets) | encodage $-12\,\%$, décodage $-10\,\%$ | gardé |
| Mise à jour d'intervalle sans branchement, isolée dans une boucle | $-30\,\%$ (2,9 ns contre 4,1 par bit) | voir ligne suivante |
| La même dans le codec, signes dans un second flux codé en parallèle | encodage $+20\,\%$ | retiré |
| Deux codeurs avancés en alternance | aucun gain | retiré |

L'écart entre les lignes 3 et 4 est instructif. Le drapeau « trit nul » décide s'il faut coder un signe : supprimer le branchement dans le codeur le fait réapparaître une ligne plus bas, avec la même erreur de prédiction. Le supprimer là aussi oblige à exécuter un pas de codeur « à vide » pour les 60 % de trits nuls, ce qui coûte plus que les erreurs de prédiction évitées. Le coût est le nombre de décisions codées ; il n'y a pas de facteur 2 à attendre d'un réglage fin. Au-delà il faudrait une autre famille de codeur (rANS à flux entrelacés et probabilités figées par paquet), qui perdrait l'essentiel du gain apporté par les contextes.

Pour que le pire cas ne dépasse jamais le budget de trame, `cfg.ec_max_blocks` plafonne le nombre de blocs codés par appel : au-delà, le reste de l'appel part en jetons, désormais très bon marché. La bascule se fait en frontière de paquet et le décodeur la suit grâce au type de paquet, donc le plafond borne aussi le décodage. Les trames calmes gardent tout le gain, les trames chargées retombent vers le débit des jetons au lieu de déborder (`test_entropy_cpu_cap`).

### 8.5 Quantification côté encodeur

En balayant le seuil, le PSNR baissait quand le seuil baissait (35,6 dB au seuil 3, 28,6 dB au seuil 1 sur la scène 720p). Cause : avec l'ADM le pas double à chaque trit non nul, et l'encodeur émettait un trit dès que l'erreur dépassait le seuil, même quand le pas était plus grand que deux fois l'erreur. La référence oscillait alors autour de la cible avec un pas croissant.

Règle actuelle : un trit non nul n'est émis que si $|e| > \theta$ et $2|e| > \delta$, c'est-à-dire seulement si le pas rapproche la référence de la source. Sinon le trit vaut $0$, ce qui divise le pas par deux ; un bloc dont un pas doit décroître est transmis même si ses dix trits sont nuls. Le décodeur et le format ne changent pas.

| Test | Avant | Après (jetons) | Après (codage entropique) |
| :--- | :--- | :--- | :--- |
| Tile MV, luma | 36,08 dB à 2,20 bpp | 40,06 dB à 1,62 bpp | 40,06 dB à 0,50 bpp |
| 720p NV12, seuil 1 | 28,6 dB | 37,1 dB | 37,1 dB |
| Shader HLSL (GPU, `--synth --psnr`) | 35,8 dB à 20,1 Mbit/s | 37,0 à 37,3 dB à 17,3 Mbit/s | sans objet |

La même règle est appliquée dans `ternary_encode.hlsl` (`Quant()`), vérifiée sur GPU avec 0 pixel d'écart entre la référence de l'encodeur et la sortie du décodeur.

### 8.6 Ordre des ancres et compensation de mouvement

La scène 3D restait à 27 dB. La cause n'était pas la modulation delta : le décalage global découvre une bande de bord remplie de gris neutre et entraîne le HUD fixe avec le fond, et le test n'appelait pas la réparation par ancres. Les deltas seuls mettent plusieurs trames à rattraper une erreur de 100 niveaux, et le décalage la recrée à chaque trame.

| Scène 3D, GMC | PSNR | Débit (jetons) | Débit (codé) |
| :--- | ---: | ---: | ---: |
| Deltas seuls | 26,89 dB | 1,04 bpp | 0,26 bpp |
| Deltas puis `repair_disocclusions` | 37,93 dB | 1,79 bpp | 0,57 bpp |

Ce diagnostic a aussi révélé une règle d'ordre. Le récepteur applique la fuite une fois, avant tout paquet ; l'encodeur l'applique à chaque échantillon visité par la passe delta. Une ancre émise avant la passe delta est donc atténuée côté encodeur seulement, et les deux références divergent dès que $\gamma < 1$ (reproduit avec `gamma_q8 = 254`). Règle : les ancres d'une trame sont émises après ses deltas. Les deux tests de scène vérifient maintenant l'égalité bit à bit des références à chaque trame.

Le décodage multi-thread respecte le même ordre : `zcrudp_ternary_mt_decode_packets()` découpe le lot là où deltas et ancres alternent et décode chaque série en parallèle (`test_mt_decode_keeps_packet_order`).

Limite restante levée (Section 9) : sur une photographie détaillée en panoramique, la combinaison de la compensation de mouvement `TileMV` (amorçage spatial + recherche logarithmique `2×2` + raffinement `±1` sur CPU et GPU `CSTileMV`), du décalage cohérent de `delta_buf` (`apply_tile_mv_state`), et de la réparation spatiale quantifiée `INTRA_Q_EC` (`cfg.intra_qstep >= 2`) élimine le plafond des 25 dB.

---

## 9. Levée de la limite 25 dB, optimisation 360 Hz et preuves formelles

### 9.1 Prédiction avancée et borne analytique universelle
* **`INTRA_Q_EC` (`ZCRUDP_TERNARY_PKT_INTRA_Q_EC`)** : prédiction spatiale JPEG-LS MED quantifiée au pas `qstep >= 2` sur le tampon reconstruit `ref`, suivie du range coder réinitialisé par paquet. Réduit le coût des ancres de réparation de ~44 % tout en bornant l'erreur ponctuelle à $\lfloor q/2 \rfloor$ pour tout couple `(src, pred)`.
* **Borne pire cas universelle** : `zcrudp_ternary_repair_disocclusions()` déclenche une ancre dès que `bad >= 2 || max_err > 2 * err_thresh`, garantissant analytiquement $\text{MSE}_{\max} \le \max(\lfloor q/2 \rfloor^2, \frac{67}{64} E^2)$ pour toute image possible ($\ge 35{,}89\text{ dB}$ pour $E=4, q=4$).
* **JND fermé arête-vs-texture (`cfg.jnd_boost`)** : `zcrudp_ternary_block_jnd_threshold()` sépare sur `ref` les arêtes alignées (UI/texte, cohérence directionnelle élevée $\rightarrow$ seuil inchangé) des textures désordonnées (énergie élevée et faible cohérence $\rightarrow$ relèvement du seuil sans dérive ni bit de signalisation).

### 9.2 Optimisations débit/latence 1080p360
* **Décompactage sans division (CPU, AVX2 16 voies, HLSL)** : factorisation 32 bits (`69043U >> 24`) et Granlund-Montgomery 16 bits (`3507`) dans `protocol_ternary.h`, `ternary_codec.h` (`zcrudp_ternary_unpack10_x16_avx2`) et `ternary_decode.hlsl`.
* **rANS statique 4 voies (`ternary_rans.h`, `DELTA_RANS`)** : 4 états `uint32_t` entrelacés sur 4 profils CDF 12 bits avec repli automatique vers `DELTA` si le paquet ne rétrécit pas.
* **Realloc v2 (`DELTA_RV2`)** : réalloue les 6 487 codes `[59049..65535]` en RLE double résolution (`1..384` + puissances de 2 `512..32768`) et fusion saut + bloc de poids 1 (`W1`) ou 2 (`W2`) sans division.
* **Flux auxiliaires & décodage fusionné** : compression `TILE_MV_DPCM`, partage `Y -> UV` en NV12 avec alignement pair strict (`apply_tile_mv_nv12_state`), décodage en une passe (`decode_packets_fused`), sommeil OS dans `ternary_mt.h`, et parallélisation `zcrudp_ternary_mt_encode_intra_range()`.

### 9.3 Suite de preuves exhaustives (`tests/test_ternary_proofs.c`)
Exécutée automatiquement par `make test` et `make sanitize` (ASan + UBSan) :
1. Descente stricte du quantificateur (`20 971 520` tuples).
2. Borne d'erreur `INTRA_Q_EC` sur tous les `4 128 768` triplets `(src, pred, q)`.
3. Parité extrême `q = 2, err = ±255` (`q_res = ±128`, `0` pixel d'écart).
4. Borne adversariale universelle ($\ge 35{,}89\text{ dB}$ sur bruit blanc, damier inversé et impulsions 64 octets).
5. Récupération `TileMV` sur 8 directions/vitesses de panoramique ($\ge 38\text{ dB}$, `0` pixel d'écart).
6. Parité décroissance AVX2 vs scalaire sur tous les `16 777 216` triplets `(px, neutral_y, gamma_q8)`.
7. Décompactage sans division sur les `59 049` mots et formule Granlund-Montgomery `/ 243` sur les `65 536` valeurs `uint16_t`.
8. Parité `DELTA_RANS` + `decode_packets_fused` et sûreté sur paquets hostiles.
9. JND fermé (`-79,4 %` octets sur texture bruitée, `0,00` d'écart MSE sur UI nette).
10. Résilience et récupération sous pertes UDP en rafale Gilbert–Elliott (`38,17 dB`).
11. Parité exhaustive `Realloc v2` (`W1`, `W2`, RLE puissances de 2, `-45,3 %` sur blocs dispersés).
12. Compression `TILE_MV_DPCM` (`-91,8 %`) et alignement NV12 `Y -> UV` (`dx = 7` impair, `0` inversion U/V).
13. Parité `INTRA_Q_EC` multi-thread (`4` threads, `0` pixel d'écart).
