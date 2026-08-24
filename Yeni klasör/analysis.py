import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

pd.set_option('display.max_columns', 20)

df = pd.read_csv('cost-of-living_v2.csv')
print(df.shape)
df.head()


rename_map = {
    "x1": "meal_inexpensive_restaurant",
    "x2": "meal_2people_midrange_restaurant",
    "x3": "mcmeal_mcdonalds",
    "x4": "domestic_beer_restaurant_05l",
    "x5": "imported_beer_restaurant_033l",
    "x6": "cappuccino_restaurant",
    "x7": "coke_pepsi_restaurant_033l",
    "x8": "water_restaurant_033l",
    "x9": "milk_1l",
    "x10": "bread_loaf_500g",
    "x11": "rice_white_1kg",
    "x12": "eggs_regular_12",
    "x13": "local_cheese_1kg",
    "x14": "chicken_fillets_1kg",
    "x15": "beef_round_1kg",
    "x16": "apples_1kg",
    "x17": "banana_1kg",
    "x18": "oranges_1kg",
    "x19": "tomato_1kg",
    "x20": "potato_1kg",
    "x21": "onion_1kg",
    "x22": "lettuce_1head",
    "x23": "water_market_15l",
    "x24": "wine_midrange_market",
    "x25": "domestic_beer_market_05l",
    "x26": "imported_beer_market_033l",
    "x27": "cigarettes_20pack_marlboro",
    "x28": "one_way_ticket_local_transport",
    "x29": "monthly_pass_transport",
    "x30": "taxi_start_normal_tariff",
    "x31": "taxi_1km_normal_tariff",
    "x32": "taxi_1hour_waiting",
    "x33": "gasoline_1l",
    "x34": "vw_golf_new_car",
    "x35": "toyota_corolla_new_car",
    "x36": "basic_utilities_85m2",
    "x37": "mobile_tariff_1min_local",
    "x38": "internet_60mbps_monthly",
    "x39": "fitness_club_monthly",
    "x40": "tennis_court_1hour_weekend",
    "x41": "cinema_1seat_international",
    "x42": "preschool_fullday_private_monthly",
    "x43": "intl_primary_school_yearly",
    "x44": "jeans_levis501",
    "x45": "summer_dress_chain_store",
    "x46": "nike_running_shoes_midrange",
    "x47": "leather_business_shoes_men",
    "x48": "rent_1bedroom_citycentre",
    "x49": "rent_1bedroom_outside_centre",
    "x50": "rent_3bedroom_citycentre",
    "x51": "rent_3bedroom_outside_centre",
    "x52": "price_per_sqm_citycentre",
    "x53": "price_per_sqm_outside_centre",
    "x54": "avg_monthly_net_salary",
    "x55": "mortgage_interest_rate_pct"
}

df = df.rename(columns=rename_map)
all_feat_cols = list(rename_map.values())
print(df.columns.tolist())


print("Toplam satır:", len(df))

miss_by_col = df[all_feat_cols].isna().mean().sort_values(ascending=False)
print("\nEn yüksek eksiklik oranına sahip sütunlar:")
print(miss_by_col.head(6))


drop_cols = ['tennis_court_1hour_weekend', 'price_per_sqm_outside_centre',
             'price_per_sqm_citycentre', 'monthly_pass_transport']
feat_cols = [c for c in all_feat_cols if c not in drop_cols]
print("Kullanılan özellik sayısı:", len(feat_cols), "(55'ten", len(drop_cols), "tanesi çıkarıldı)")


row_miss = df[feat_cols].isna().mean(axis=1)
dfc = df[row_miss <= 0.30].reset_index(drop=True)

print(f"Filtre sonrası satır sayısı: {len(dfc)} / {len(df)} (%{len(dfc)/len(df)*100:.0f})")
print("\nBu altkümede data_quality dağılımı (bilgi amaçlı):")
print(dfc['data_quality'].value_counts())
print("\n-> Önceki 923 satıra kıyasla ~4.5 kat daha fazla veri kullanılıyor.")


print("Örnek - winsorize öncesi en yüksek internet ücreti:",
      dfc['internet_60mbps_monthly'].max())

for c in feat_cols:
    lo, hi = dfc[c].quantile([0.01, 0.99])
    dfc[c] = dfc[c].clip(lo, hi)

print("Winsorize sonrası en yüksek internet ücreti:",
      round(dfc['internet_60mbps_monthly'].max(), 1))


X = dfc[feat_cols].copy()

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

print("Ölçeklenmiş veri boyutu:", X_scaled.shape)


pca_full = PCA(n_components=0.90, random_state=42)
X_pca = pca_full.fit_transform(X_scaled)

print("Varyansın %90'ını açıklamak için gereken bileşen sayısı:", X_pca.shape[1])

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(7,4))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.axhline(0.90, color='red', linestyle='--', label='%90 varyans')
plt.xlabel('Bileşen sayısı')
plt.ylabel('Kümülatif açıklanan varyans')
plt.title('PCA - Açıklanan Varyans')
plt.legend()
plt.tight_layout()
plt.show()


loadings = pd.DataFrame(pca_full.components_[:2].T, index=feat_cols, columns=['PC1', 'PC2'])

print("PC1'e en çok katkı veren değişkenler (mutlak değer):")
print(loadings['PC1'].abs().sort_values(ascending=False).head(8))

print("\nPC2'ye en çok katkı veren değişkenler (mutlak değer):")
print(loadings['PC2'].abs().sort_values(ascending=False).head(8))

# Yorum: PC1 -> restoran/market fiyatları gibi genel fiyat seviyesi ekseni.
# PC2 -> giyim/benzin/market ürünleri ağırlıklı, ikincil bir tüketim-kalıbı ekseni.


ks = list(range(2, 8))
wcss = []
sils = []
for k in ks:
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_test = km_test.fit_predict(X_pca)
    wcss.append(km_test.inertia_)
    sils.append(silhouette_score(X_pca, labels_test))
    print(f"k={k}  WCSS={km_test.inertia_:.1f}  silhouette={sils[-1]:.3f}")

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(ks, wcss, marker='o', color='#4C72B0', label='WCSS (Inertia)')
ax1.set_xlabel('Küme sayısı (k)')
ax1.set_ylabel('WCSS (Inertia)', color='#4C72B0')
ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax1.axvline(3, color='gray', linestyle=':', alpha=0.7)

ax2 = ax1.twinx()
ax2.plot(ks, sils, marker='s', color='#C44E52', label='Silhouette')
ax2.set_ylabel('Silhouette Skoru', color='#C44E52')
ax2.tick_params(axis='y', labelcolor='#C44E52')

plt.title(f'Elbow (WCSS) ve Silhouette Skorları - k Seçimi (n={len(dfc)})')
fig.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_pca)
dfc['cluster'] = labels

print(dfc['cluster'].value_counts().sort_index())
print("\nSeçilen k=3 için silhouette skoru:", round(silhouette_score(X_pca, labels), 3))


key_vars = ['meal_inexpensive_restaurant', 'meal_2people_midrange_restaurant', 'gasoline_1l',
            'basic_utilities_85m2', 'rent_1bedroom_citycentre', 'rent_1bedroom_outside_centre',
            'avg_monthly_net_salary', 'cigarettes_20pack_marlboro', 'internet_60mbps_monthly']
summary_tmp = dfc.groupby('cluster')[key_vars].mean()

# Kümeleri ortalama maaşa göre otomatik etiketle
order = summary_tmp['avg_monthly_net_salary'].sort_values(ascending=False).index.tolist()
names = {order[0]: 'Yüksek gelir', order[1]: 'Orta-yüksek gelir', order[2]: 'Düşük maliyet'}
colors = ['#4C72B0', '#DD8452', '#55A868']

tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
X_tsne = tsne.fit_transform(X_pca)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

for c in range(3):
    mask = dfc['cluster'] == c
    lbl = f"{c} - {names[c]}"
    ax[0].scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, c=colors[c], label=lbl, alpha=0.5)
ax[0].set_xlabel('PC1'); ax[0].set_ylabel('PC2')
ax[0].set_title(f'PCA (PC1 vs PC2) ile Kümeler (k=3, n={len(dfc)})')
ax[0].legend(fontsize=8)

for c in range(3):
    mask = dfc['cluster'] == c
    lbl = f"{c} - {names[c]}"
    ax[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=8, c=colors[c], label=lbl, alpha=0.5)
ax[1].set_xlabel('t-SNE 1'); ax[1].set_ylabel('t-SNE 2')
ax[1].set_title('t-SNE ile Kümeler (k=3)')
ax[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('clusters_pca_tsne.png', dpi=150)
plt.show()


summary = dfc.groupby('cluster')[key_vars].mean().round(1)
summary.T


profile_cols = feat_cols
means = dfc.groupby('cluster')[profile_cols].mean()
means_z = (means - means.mean()) / means.std()

fig, ax = plt.subplots(figsize=(9, 13))
im = ax.imshow(means_z.T.values, cmap='RdBu_r', aspect='auto', vmin=-1.5, vmax=1.5)
ax.set_xticks(range(3))
ax.set_xticklabels([f'Küme {c}' for c in range(3)])
ax.set_yticks(range(len(profile_cols)))
ax.set_yticklabels(profile_cols, fontsize=7)
plt.colorbar(im, label='Standardize edilmiş ortalama (z-score)')
plt.title(f'Küme Profilleri - Tüm Değişkenler (n={len(dfc)})')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=150)
plt.show()


for c in range(3):
    sub = dfc[dfc['cluster'] == c]
    print(f"\n=== Küme {c} - {names[c]} (n={len(sub)}) ===")
    print("En çok görülen ülkeler:")
    print(sub['country'].value_counts().head(6))
    print("Örnek şehirler:", sub['city'].sample(min(8, len(sub)), random_state=1).tolist())


dfc.to_csv('cost_of_living_clustered.csv', index=False)
print(f"Kaydedildi: cost_of_living_clustered.csv ({len(dfc)} satır)")
