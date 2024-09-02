# Synthetic Time series Generation 

**Projekt kerete: Önálló laboratórium**

**Riport és projekt készítője: Török Ákos - H8YDYG**

**Konzulens: Ács Gergely**

## 1 Bevezetés 
A szintetikus adatok generálása fontos része a kutatásoknak és a fejlesztéseknek. Minél több adatunk van, annál több információt tudunk kinyerni belőlük, melyeket felhasználva komoly fejlesztések mehetnek végbe.
A projekt során eredeti rögzített CTG (kardiotokográfia) adatokból generálunk szintetikus adatot. A CTG adatok FHR (Fetal Heart Rate - magzat szívritmus) és UC (Uterine Contractions - méhösszehúzódás) értékekből álló idősorok, ezen utóbbit nem vettem figyelembe az eredmény kiértékelése közben. A feladat célja, hogy az orvosi egyetemeknek legyen elég adata betanítani az orvostanhallgatókat a generált idősorok segítségével, de úgy, az adatok reprezentálják az eredeti adat jellemzőit, megfelelő minőségűek legyenek, illetve ne lehessen a szintetikus adatot konkrét személyekhez kötni, akiknek az adataiból generáltunk. 

### 1.2 Megoldás ismertetése
A projekt megvalósítása során először a megszerzett adatokat megfelelően kell kezelni. Ezeken az adatokon történik a generatív modellek betanítása, amelyek általában Generative Adversarial Network (GAN) neurális háló alapon működnek, ezek segítségével generálhatunk több, eredeti idősorokhoz hasonlító szintetikusat. Ezt követően összevetjük az eredeti adattal az elkészült szintetikus adatunkat, és lemérjük a hasonlóságot. A cél minél reprezentatívabb, realisztikusabb adat kinyerése.

## 2 Adatok előkészítése 

Az eredeti adatban gyakran előfordult, hogy huzamosabb ideig vagy akár rövid kihagyásokkal hiányzik a jel. A csv fájlokat rekurzív felezéssel interpoláltam egy adott thresholdig, ami az egymást követő nulla értékek számát jelenti. Ezt követően pedig eldaraboltam a fájlokat a megmaradt nulla sorozatok mentén, és kiválogattam a 60 szekundumnál nagyobbakat. Így az eredeti 552 idősorunkból 7929 idősor lett. Ebből a halmazból véletlenszerűen lett kiválogatva 552 darab idősor, amelyeken történt a további munka idő- és hardverlimitációból adódóan.

## 3 Kipróbált modellek 

### 3.1 Y-data

A python csomag ingyenesen telepíthető vagy letölthető a program [github](https://github.com/ydataai/ydata-synthetic) oldaláról. A pip telepítés során nem az akkori legfrissebb verziót (1.3.3 2024-01-08) telepítette, így manuális letöltés után a setup.py-t futtatva sikerült a friss csomagot felrakni. A program két fajta modellel használható szintetikus time series generálására, ezek a TimeGan és a DoppelGANger.

#### 3.1.2 TimeGan: 

A modell ablakokra osztja be a tanulandó adatot, így leegyszerűsítve sajátítja el az idősor egy tipikus jellemzőjét a tanulás során. Az adatokat nyersen kapja meg, majd automatikusan normalizálja, azonban így is kapjuk meg, így visszaalakításra van szükség.

Részlet a TimeGan modellel generált adatokból visszaalakítva: 

    FHR,UC 
    124.97963247406251,0.46213 
    126.538649190625,0.40625685 
    127.53660154781251,0.34827533 
    127.96714688218749,0.29868218 
    128.030657240625,0.26060945 
    127.90405288125,0.23302533 
    127.69854425,0.21361725 
    127.475453565625,0.20015846 
    127.266043845625,0.19089729 
    127.08466584687498,0.18455732 
    126.93593118125,0.18023665
    126.81890983281251,0.17730556 
    126.729675821875,0.175327 
    126.66325096093752,0.17399856 
    126.61477699156251,0.17311166 
    126.5800880759375,0.17252284 
    126.5558239375,0.17213392 
    126.53937329062501,0.171878 
    126.528647559375,0.17171004 

A generált adatokból látható, hogy mindig egy bizonyos érték körül sorsol, ez az érték megegyezik nagyjából a medián értékkel. Az értékek periodikusan ismétlődnek. A következő ábrán egy szintetikus idősor látható kirajzolva, melyen jól láthatóak az előbb említett karakterisztikák.

![Y-data generált time series](ydata_timeseries.png)

TimeGan-nel generált szintetikus adatok gyakorisága összevetve az eredetivel: 

![Y-data összevetve az eredetivel](ydata_tgan_compare.png)
 

#### 3.1.2 Dgan (DoppelGANger): 

A modell Generative Adversarial Network (GAN) framework-öt használja idősorok generálásához, mely során a tanulandó adat időbeli függőségeiből és karakterisztikáiból tanul. Főként tabular data-hoz ajánlott, de time series-hez is használható, azonban időbélyegek híján nem bizonyult használhatónak. A méréseknél kizárólag a TimeGan által generált idősorokat vettem figyelembe.

Részlet a Dgan modellel generált szintetikus idősorokból:

    1.9827234745025635,87.5
    1.9493281841278076,87.5
    1.99649977684021,87.5
    1.996471881866455,87.5
    0.7524802684783936,87.5
    0.0,87.5
    0.0,87.5
    0.0,87.5
    0.0,87.5
    0.0,87.5
    0.0,87.5

### 3.2 Gretel ai 

A gretel csomag szintén a doppelganger modellt használja tanuláshoz. A telepítéshez elegendő volt a python csomag managert használni. A forráskód megtalálható a program [github](https://github.com/gretelai/gretel-python-client) oldalán is A gretel-client csomag minden formációban felhasználja az api kulcsunkat a használathoz, így korlátozott mind lokális dockerben mind a gretel felhőben való használatban.

gretel_gen0.csv részlet: 

    ,FHR,UC,example_id 
    0,129.251953125,50.80496597290039,0 
    1,103.07565307617188,45.42721557617188,0 
    2,101.42813873291016,34.90209197998047,0 
    3,105.33518981933594,33.037200927734375,0 
    4,110.35237121582033,29.6757755279541,0 
    5,104.24378967285156,32.50171661376953,0 
    6,105.33545684814452,27.085033416748047,0 
    7,100.98148345947266,21.410734176635746,0 
    8,94.27935028076172,20.28357696533203,0 
    9,92.1179428100586,18.194080352783203,0 
    10,114.45184326171876,57.43331527709961,1 

Példa a generált idősorok típusára kirajzolva:

![Gretel-ai generált time series](gretel_timeseries.png)

Az ábra alapján látszik, hogy a modell egy ugráló tendenciát sajátított el az eredeti adat alapján, melynek következtében spike szerű idősorokat kapunk.

A generált értékek gyakorisága az eredetihez mérten egészen hasonló.

![Gretel-ai értékek eloszlása összevetve az eredetivel](gretel_compare.png)

Gretel által generált szintetikus és eredeti értékek frekvenciája. 

### 3.3 Deep echo 

A Deep echo a [Synthetic Data Vault](https://sdv.dev/) csomag része, azonban önállóan is lehet használni. A framework a PARModel-t használja idősorok tanulásához és generálásához, ami egy Probabilistic AutoRegressive model, amely deep learning segítségével tanul. Telepíthető pip segítségével, vagy letölthető [github](https://github.com/sdv-dev/DeepEcho/tree/main)-ról szintén.

A sorsolt értékek mindig egy adott konstanshoz térnek vissza, gyakran egymás után ismétlődve, főként ezen értéktől lentebb sorsolva mozognak.

![Deep echo time series](deep_echo_timeseries.png)

A szintetikus értékek gyakoriságát összemérve az erdetivel jól látható, hogy nagyjából a medián érték körül sorsol.

![Deep echo eloszlás összevetve az eredetivel](deep_echo_compare.png)
 
### 4 Összehasonlítás 

Az összehasonlításhoz egy 552 fájlból álló random adatbázist ragadtam ki az eredeti adatbázisból. A generált adatbázis szintén 552 fájlból áll, melyek hosszúsága a referencia hosszakból kerültek kiválasztásra randomizálva. Ezeken az idősorokon történtek a metrikák alkalmazása.

### 4.1 Jensen-Shannon divergencia 

A [Jensen-Shannon divergencia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) értékek valószínűségi eloszlásának hasonlóságára alkalmas metrika time series-ek esetében. Más néven *information radius* vagy *total divergence to the average*. Az értékeket 100 csoportra osztva számoltam ki a divergenciát, azaz nagyjából 2 értéktartományonként. A szimilaritás annál nagyobb mértékű, minél jobban közelít a 0 értékhez.

Y-data: 0.7741785821804532

Gretel ai​: 0.15914649623590504

Deep echo​: 0.43564005217707696

### 4.2 Euklidészi távolság

Az [euklidészi távolság](https://en.wikipedia.org/wiki/Euclidean_distance) megadja két pont közti távolságot, ami a köztük lévő vonalszakasz hossza. Gyakran nevezik pitagorasz távolságnak. Hasonlóan mérhetjük vele a szimilaritás nagyságát idősorok esetében. Az idősorok spektrumát diszkrét koszinusz transzformációval ([DCT](https://hu.wikipedia.org/wiki/Diszkr%C3%A9t_koszinusz-transzform%C3%A1ci%C3%B3#:~:text=A%20diszkr%C3%A9t%20koszinusz%2Dtranszform%C3%A1ci%C3%B3%20(angol,komplex%2C%20hanem%20val%C3%B3s%20sz%C3%A1mokon%20dolgozik.&text=A%20t%C3%B6m%C3%B6r%C3%ADt%C3%A9s%20sor%C3%A1n%20a%20l%C3%A9nyegtelen%20inform%C3%A1ci%C3%B3k%20elhagy%C3%A1s%C3%A1ra%20t%C3%B6reksz%C3%BCnk.))) számoltam ki, ezekből pedig átlagoltam a spektrumokat mind a két halmazon. Ezek távolságai és a spektrumok abszolút különbségei az alábbiak:

    Y-data: 7.801970566407013

![Eredeti spektrum](eredeti_spectrum1.png)
![Y-data spektrum](ydata_spectrum2.png)
![Spektrumok különbségei](ydata_spectrum3.png)

    Gretel-ai: 6.928485721164687

![Eredeti spektrum](eredeti_spectrum1.png)
![Gretel spektrum](gretel_spectrum2.png)
![Spektrumok különbségei](gretel_spectrum3.png)

    Deep-echo: 1.6311129882389055

![Eredeti spektrum](eredeti_spectrum1.png)
![Deep echo spektrum](deep_spectrum2.png)
![Spektrumok különbségei](deep_spectrum3.png)

### 4.3 Összesítve

Csomag | Y-data | Gretel-ai | Deep-echo |
--- | --- | --- | --- |
Jensen-Shannon div.| 0.7741785821804532 | 0.15914649623590504 | 0.43564005217707696 |
Euklidészi távolság | 7.801970566407013 | 6.928485721164687 | 1.6311129882389055 |

## 5 Saját eredmény értékelése

### 5.1 Elsajátított tudás

A projekt során megismerkedtem a generatív modellek felépítésével és működésükkel. Ezen felül az adatok megfelelő válogatásába és kezelésébe, például az adatok szűrése, ezek mennyiségének szakszerű definiálásába nyertem betekintést. Továbbá az idősorokhoz köthető metrikák és azok alkalmazásairól - például értékek eloszlása, ezek alapján számolt divergencia - szereztem ismeretet.

### 5.2 Nem várt nehézségek

Némelyik csomag nem települt fel megfelelően az ezt hivatott tool által, majd ezt manuálisan kellett javítani. A generatív programok használata első ránézésre bonyolultnak tűntek, majd az eredmények is érthetetlennek bizonyultak első tesztelések során, például a Y-data DGan modell esetében, az idősorokhoz szánt példakód tabular data-ra voltak optimalizálva.

### 5.3 Önértékelés

A projekt keretében az adatok megfelelő interpolációját kellett többször megkísérelni, mivel az első próbálkozások után sem volt megfelelő eredmény e tekintetben, végül egy alkalmas technikával sikerült orvosolni. Legjobban az adatok vizualizációját, az eredmény átvitt értelemben értett életre keltését élveztem. A modelleken és eredményeken végtelen mértékig lehet javítani, csiszolni, illetve elegendő erősséggű munkaeszköz beszerzése is ajánlott.

## 6 Függelék

### 6.1 Interpolálás

A [nyers adatok](https://github.com/anantgupta129/CTU-CHB-Intrapartum-Cardiotocography-Caesarean-Section-Prediction/tree/main/database/signals) feldolgozásához használt interpoláció a rekurzív felezés volt.

Részlet az adatok_interpolalas.py fájlból: 
    
    def recursive_halfing_interpolation(start_idx, end_idx, fhr_values): 

        try: 
            num_zeros = end_idx - start_idx - 1 #nullak szama a szelso indexek kulonbsege 
            if num_zeros >= 1:  
                #a kozepso index az indexek atlaga 
                middle_idx = (start_idx + end_idx) // 2  
                if fhr_values[middle_idx] == 0: #ha nulla a kozepso 
                    left_adjacent = fhr_values[start_idx] #bal oldai ertek 
                    right_adjacent = fhr_values[end_idx] #jobb oldali ertek 
                    middle_value = (left_adjacent + right_adjacent) / 2  #a ket ertek atlaga 
                    fhr_values[middle_idx] = middle_value   #lesz a kozepso ertek 
    
                #meghivja kozeptol balra es jobbra mindket iranyban
                recursive_halfing_interpolation(start_idx, middle_idx, fhr_values) 
                recursive_halfing_interpolation(middle_idx, end_idx, fhr_values) 

        except IndexError: 
                    pass 

### 6.2 Csomagok használata

A csomagokhoz tartoznak útmutatók, a saját verzió csupán kis mértékben tér el a példakódoktól, az eredeti leírás elegendő a használathoz. A [Y-data](https://github.com/ydataai/ydata-synthetic/blob/dev/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb), [Gretel](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/local_synthetics.ipynb), és a [Deep-echo](https://github.com/sdv-dev/DeepEcho/blob/main/tutorials/01_Getting_Started.ipynb) útmutatója a linkekre kattintva megtalálhatóak (*linkek aktívan elérhetőek: 2024. 05. 30.*).

A csomagok értékelése hasznáalti szempontból:

Csomag | Y-data | Gretel-ai | Deep-echo |
--- | --- | --- | --- |
Fizetős| részben | igen | nem |
Normalizálás | végeredmény manuális | auto | auto | 
Gyorsaság | átlagos | gyors (felhő) | átlagos | 

### 6.3 Összehasonlításhoz használt kódok

#### 6.3.1 Jensen-Shannon divergencia
    #jensen shannon div histogrambol
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import jensenshannon

    input_file = "generalt_idosorok"
    input_file2 = "eredeti_idosorok"
    data = pd.read_csv(input_file)
    data2 = pd.read_csv(input_file2)
    min_value = data2['FHR'].min() #a ket szelsoerteket az eredeti idosorokbol adjuk meg
    max_value = data2['FHR'].max()
    bin_edges = np.linspace(min_value, max_value, 100 + 1) #100 reszre osztjuk az ertekeket, ezt lehet modositani
    #letrehozunk ket histogramot normalizalva (density)
    hist1, _ = np.histogram(data['FHR'], bins=bin_edges, weights=data['Relative Frequency'], density=True) 
    hist2, _ = np.histogram(data2['FHR'], bins=bin_edges, weights=data2['Relative Frequency'], density=True)
    js_divergence = jensenshannon(hist1, hist2) #divergencia szamitasa scipy csomaggal

    print(f"Jensen-Shannon Divergence: {js_divergence}")

#### 6.3.2 Diszkrét koszinusz transzformáció, spektrumok és euklidészi távolság

**Spektrumok kiszámítása:**

    from scipy.fftpack import fft, dct, idct
    from sklearn.preprocessing import MinMaxScaler

    source_folder0 = "eredeti_idosor"
    source_folder1 = "generalt_idosor"
    def mean_spectrum(source_folder, name):
        dct_coefficients = [] #ebbe kerulnek a transzformalt ertekek
        for file_name in os.listdir(source_folder):
            if file_name.endswith('.csv'):
            
                file_path = os.path.join(source_folder, file_name)
                data = pd.read_csv(file_path)
                column_data = data['FHR'].values #csak az frh ertekek
                
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_normalized = scaler.fit_transform(column_data.reshape(-1, 1)).flatten() #normalizacio

                fixed_length = 500 #az ertekek hossza
                #fontos, hogy ugyan olyan hosszuak legyenek
                if len(data_normalized) < fixed_length:
                    data_normalized = np.pad(data_normalized, (0, fixed_length - len(data_normalized)), 'constant')
                else:
                    data_normalized = data_normalized[:fixed_length]
                dct_transformed = dct(data_normalized, norm='ortho')
                dct_coefficients.append(dct_transformed) #hozzaadjuk a kapott ertekeket

        dct_coefficients = np.array(dct_coefficients)
        mean_dct_spectrum = np.mean(dct_coefficients, axis=0) #a spektrum atlagolasa
        #kirajzolas
        plt.figure(figsize=(10, 2))
        plt.plot(mean_dct_spectrum[1:], linewidth=2)
        plt.title(f'{name} átlag spektruma')
        plt.xlabel('Index')
        plt.ylabel('Átlag DCT együttható')
        plt.show()

        return mean_dct_spectrum
        
**Példa használat:**
    eredeti_spectrum = mean_spectrum(source_folder0, "Eredeti adatok")    
    generalt_spectrum = mean_spectrum(source_folder1, "Generált adatok")
    kulonbseg = abs(eredeti_spectrum - generalt_spectrum)

**Euklidészi távolság:**
    import numpy as np
    #a dct utan kapott adatokat kell megadnunk
    dist = np.linalg.norm(eredeti_spectrum-generalt_spectrum)
    print(f"euklidészi távolság: {dist}")
