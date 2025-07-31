# Νικόλαος Ζομπάς, 4169
# Εριφύλη Λουκοπούλου, 4196
# Το άθροισμα των ΑΕΜ μας είναι μονός αριθμός

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

#------------------------------------------------------------------------------------------------------------------------------------------------- 
# Visualization Of 1378 Data

# Διάβασμα του αρχείου με τα δεδομένα για χρήση και αποθήκευση τους στον πίνακα t
data=pd.read_csv("data.csv")
t= data[['X','Y']].values

# Αναπαράσταση των δεδομένων πριν την εφαρμογή των αλγορίθμων
plt.figure(num="Data Visualization")
plt.scatter(t[:, 0], t[:, 1], s=10, color='blue')
plt.title('Visualization of Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Silhouette Score

# Εφαρμογή του συντελεστή σιλουέτας για να βρεθεί ο αριθμός των συστάδων χρησιμοποιόντας K-means
silhouette_scores = [] # Λίστα που θα περιέχει τα αποτελέσματα του συντελεστή σιλουέτας

# Υπολογισμός του σκορ για 2,3,4,...,19,20 συστάδες 
for n in range(2, 20):
    # Εφαρμογή του K-means
    kmeans = KMeans(n, random_state=42).fit(t)
    # Υπολογισμός του συντελεστή σιλουέτας για τον τρέχοντα αριθμό συστάδων
    sil_score = silhouette_score(t, kmeans.labels_)
    silhouette_scores.append(sil_score)

# Αναπαράσταση των αποτελεσμάτων του συντελεστή σιλουέτας για τους αριθμούς των συστάδων
plt.figure(num="Silhouette Score")
plt.plot(range(2, 20), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title("Silhouette Score")
plt.xlabel("Αριθμός Συστάδων")
plt.ylabel("Silhouette Score")
plt.show()

# O αριθμός των συστάδων με τον υψηλότερο συντελεστή σιλουέτας είναι αυτός που επιλέγεται
clusters = np.argmax(silhouette_scores) + 2 # Προσθέτoυμε 2 γιατί παίρνουμε τον δείκτη από 0 έως 18
print(f"Ο βέλτιστος αριθμός συστάδων είναι {clusters} με σκορ σιλουέτας {max(silhouette_scores,)}")

#-------------------------------------------------------------------------------------------------------------------------------------------------
# CPCC

# Υπολογισμός CPCC των ιεραρχικών αλγορίθμων (single-link και complete-link)
cpcc_scores = {} # Λίστα που θα περιέχει τα αποτελέσματα του CPCC

for i in ['single', 'complete']:
    linkage_matrix = linkage(t, i) # Εφαρμογή του τρέχοντα ιεραρχικού αλγορίθμου
    cpcc, _ = cophenet(linkage_matrix, pdist(t)) # Υπολογισμός του cpcc
    cpcc_scores[i] = cpcc

# Επιλογή του ιεραρχικού αλγορίθμου με το υψηλότερο CPCC
hierarchical = max(cpcc_scores, key=cpcc_scores.get)
print(f"Το CPCC για το single-link είναι {cpcc_scores['single']:.4f}")
print(f"Το CPCC για το complete-link είναι {cpcc_scores['complete']:.4f}")
print(f"Ο καλύτερος ιεραρχικός αλγόριθμος με βάση το CPCC είναι ο {hierarchical}-link")

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Dendrogram

# Δημιουργία και αναπαράσταση δενδρογράμματος
linkage_matrix = linkage(t, hierarchical) # Πίνακας που περιέχει τις πληροφορίες της ιεραρχικής συσταδοποίησης
plt.figure(num="Dendrogram")
dendrogram(linkage_matrix) # Δημιουργία δενδρογράμματος 
plt.title(f'Dendrogram Of {hierarchical.capitalize()}-Link')
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Hierarchical Clustering

# Εφαρμογή του ιεραρχικού αλγορίθμου που έχει προκύψει με βάση το CPCC με τόσες συστάδες όσες έχουν υπολογιστεί με βάση τον συντελεστή σιλουέτας
hierarchical_matrix = fcluster(linkage_matrix, clusters, 'maxclust') # Πίνακας που περιλαμβάνει τα δεδομένα και την συστάδα στην οποία ανήκουν

# Υπολογισμός των κέντρων των συστάδων που προέκυψαν απο τον ιεραρχικό αλγόριθμο
hierarchical_centroids = [] # Λίστα που θα περιέχει τα κέντρα
for i in range(1, clusters + 1):
    data_per_cluster = t[hierarchical_matrix == i]  # Αποθήκευση των δεδομένων της συγκεκριμένης συστάδας
    centroid = np.mean(data_per_cluster, 0)  # Υπολογισμός μέσου όρου των δεδομένων της συστάδας
    hierarchical_centroids.append(centroid)
hierarchical_centroids = np.array(hierarchical_centroids)

# Αναπαράσταση των συστάδων με βάση τον ιεραρχικό αλγόριθμο
plt.figure(num="Hierarchical Clustering")
plt.scatter(t[:, 0], t[:, 1], 10, hierarchical_matrix)
plt.scatter(hierarchical_centroids[:, 0], hierarchical_centroids[:, 1], color='red', s=80, label='Centroids')
plt.title(f"{hierarchical.capitalize()}-Link")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------
# K-means

# Εφαρμογή του αλγορίθμου τμηματοποίησης (K-means) με τόσες συστάδες όσες έχουν υπολογιστεί με βάση τον συντελεστή σιλουέτας και
# με αρχικά κέντρα, τα κέντρα της ιεραρχικής ομαδοποίηση
kmeans = KMeans(n_clusters=clusters, init=hierarchical_centroids, n_init=1).fit(t)
kmeans_labels = kmeans.labels_# Πίνακας που δείχνει σε ποια συστάδα ανήκει το κάθε δεδομένο
centroids = kmeans.cluster_centers_# Υπολογισμός των κέντρων των συστάδων που προέκυψαν από τον K-means

# Αναπαράσταση των συστάδων με βάση τον K-means
plt.figure(num="Κ-means")
plt.scatter(t[:, 0], t[:, 1], 10, kmeans_labels)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=80, label='Centroids')
plt.title("K-means")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Υπολογισμός των outliers χρησιμοποιόντας τρεις μεθόδους
#-------------------------------------------------------------------------------------------------------------------------------------------------

# Υπολογισμός των outliers με βάση μία ελάχιστη απόσταση μεταξύ των δεδομένων
distance_data_matrix = pairwise_distances(t)# Υπολογισμός του πίνακα αποστάσεων μεταξύ όλων των δεδομένων
np.fill_diagonal(distance_data_matrix, np.inf)  # Ορισμός της διαγωνίου σε άπειρο για να αγνοηθεί
min_distances = distance_data_matrix.min(axis=1) # Υπολογισμός της ελάχιστης απόστασης από κάθε δεδομένο προς τα υπόλοιπα
outliers_by_min_distance = np.where(min_distances > 10)[0]# Αν η ελάχιστη απόσταση ενός δεδομένου προς οποιοδήποτε άλλο, είναι μεγαλύτερη από 10, τότε θεωρείται outlier  

#--------------------------------------------------------------------------------------------------------------------------------------------------

# Υπολογισμός των outliers με βάση την απόστασή τους από τα κέντρα των συστάδων
distance_centroids_matrix = pairwise_distances(t, centroids) # Υπολογιμός του πίνακα αποστάσεων μεταξύ των δεδομένων και των κέντρων των συστάδων
data_centroid_distance = np.array([distance_centroids_matrix[i, cluster] for i, cluster in enumerate(kmeans.labels_)]) # Υπολογισμός της απόστασης των δεδομένων από το κέντρο της συστάδας στην οποία ανήκουν
outliers_by_centroid = np.where(data_centroid_distance > 59)[0] # Αν η απόσταση ενός δεδομένου προς το κέντρο της συστάδας του, είναι μεγαλύτερη από 59, τότε θεωρείται outlier

#--------------------------------------------------------------------------------------------------------------------------------------------------

# Υπολογισμός των outliers με βάση την πυκνότητα (DBSCAN)
# Κανονικοποίηση των δεδομένων 
S_scaler = StandardScaler()
t_scaler = S_scaler.fit_transform(t)
# Εφαρμογή του DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=10).fit(t_scaler)
dbscan_labels = dbscan.labels_ # Τα labels που προέκυψαν (-1 αντιστοιχεί σε outliers)
outliers_by_dbscan = np.where(dbscan_labels == -1)[0] # Αν το label του δεδομένου είναι -1, τότε θεωρείται outlier

#---------------------------------------------------------------------------------------------------------------------------------------------------

# Τελικός υπολογισμός των outliers με βάση τον συνδυασμό των τριών μεθόδων για την εύρεση outliers 
# (όσα είναι outliers από τουλάχιστον μία από τις τρεις μεθόδους)
outliers = np.unique(np.concatenate([outliers_by_centroid, outliers_by_min_distance, outliers_by_dbscan]))

#---------------------------------------------------------------------------------------------------------------------------------------------------

# Αναπαράσταση των outliers
plt.figure(num="Outliers")
plt.scatter(t[:, 0], t[:, 1], c=kmeans.labels_, cmap='viridis', s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', s=80, label='Centroids')
plt.scatter(t[outliers, 0], t[outliers, 1], color='red', s=10, label='Outliers')
plt.title("Outliers")
plt.legend()
plt.show()