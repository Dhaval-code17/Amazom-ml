# PERSON A — INITIAL EXPLORATORY DATA ANALYSIS (EDA) REPORT

## 1. Row Counts
- **Source 1 rows**: 2,206,821
- **Source 2 rows**: 5,034,616
- **Source 3 rows**: 5,285,603
- **Ground Truth rows**: 2,206,821

## 2. Field Names & Data Types
- **Source 1 columns**: ['entity_id', 'business_name', 'business_address', 'country']
- **Source 2 columns**: ['entity_id', 'business_name', 'business_address', 'country']
- **Source 3 columns**: ['entity_id', 'business_name', 'business_address', 'country']
- **Ground Truth columns**: ['source1_entity_id', 'matched_entity_ids']

## 3. Raw Missingness Rates
### Source 1
- `business_name` missing: 0 (0.00%)
- `business_address` missing: 0 (0.00%)
- `country` missing: 0 (0.00%)

### Source 2
- `business_name` missing: 2 (0.00%)
- `business_address` missing: 168,967 (3.36%)
- `country` missing: 0 (0.00%)

### Source 3
- `business_name` missing: 13 (0.00%)
- `business_address` missing: 175,916 (3.33%)
- `country` missing: 0 (0.00%)

## 4. Unique Entity Counts
- **Unique S1 IDs**: 2,206,821
- **Unique S2 IDs**: 5,034,616
- **Unique S3 IDs**: 5,285,603

## 5. Ground Truth Match Distribution
- **Zero matches (singletons)**: 123,247 (5.58%)
- **Exact 1 match**: 119,157 (5.40%)
- **Multiple matches (>1)**: 1,964,417 (89.02%)
- **One-to-one vs One-to-many ratio**: 119157:1964417 (0.06:1)

### Detailed Match Count Breakdown:
- `0` matches: 123,247 entities (5.58%)
- `1` matches: 119,157 entities (5.40%)
- `2` matches: 375,212 entities (17.00%)
- `3` matches: 530,841 entities (24.05%)
- `4` matches: 484,115 entities (21.94%)
- `5` matches: 321,957 entities (14.59%)
- `6` matches: 164,868 entities (7.47%)
- `7` matches: 63,968 entities (2.90%)
- `8` matches: 18,680 entities (0.85%)
- `9` matches: 4,205 entities (0.19%)
- `10` matches: 534 entities (0.02%)
- `11` matches: 37 entities (0.00%)

## 6. Encoding / Language Inspection
Lightweight sampling analysis of raw text fields for Unicode, non-ASCII rates, and script characteristics:

### Source 1 (Sample size: 50,000)
- **Countries represented**: `['US', 'India']`
- **`business_name` non-ASCII row rate**: 0 rows (0.00%)
- **`business_address` non-ASCII row rate**: 13 rows (0.03%)

### Source 2 (Sample size: 50,000)
- **Countries represented**: `['US', 'India']`
- **`business_name` non-ASCII row rate**: 7,522 rows (15.04%)
- **`business_address` non-ASCII row rate**: 4,797 rows (9.59%)
- **Sample non-ASCII business names**: `['Oyola Líberty Harmony Associates', 'ग्रेट टेक केयर प्रा. लि.', 'Stephenson, Anderson and Hill Money Ínc']`

### Source 3 (Sample size: 50,000)
- **Countries represented**: `['India', 'US']`
- **`business_name` non-ASCII row rate**: 5,772 rows (11.54%)
- **`business_address` non-ASCII row rate**: 4,575 rows (9.15%)
- **Sample non-ASCII business names**: `['Vanguard Métropolitan Sunshine', 'रियल एग्रो प्राइवेट लिमिटेड', 'Pioneer Private Límited Services']`

### Pipeline Verification:
- **Transliteration necessity**: Confirmed. Non-ASCII diacritics and accented characters (e.g. `Café`, `Résumé`, `é`, `ñ`, `ü`) exist in raw sources.
- **Normalization impact**: `unidecode` transliteration cleanly converts all non-ASCII unicode characters into plain ASCII equivalents for exact blocking and phonetic indexing.

## 7. Normalization Output Sanity Check (Sample)
### Sample 1 (ID: S1-925783039)
- **Raw Name**: `Orelee's Barbershop`
- **Normalized Name**: `orelee s barbershop`
- **Aggressive Normalized Name**: `orelee s barbershop`
- **Name Tokens**: `['orelee', 's', 'barbershop']`
- **Phonetic**: `ARLSPRPRXP`
- **Raw Address**: `1795 Westchester Drive, High Point, NC`
- **Normalized Address**: `1795 westchester drive high point nc`
- **Postal Code Guess**: ``
- **City Guess**: `point`
- **Street Number Guess**: `1795`
- **Country**: `US`

### Sample 2 (ID: S1-773889195)
- **Raw Name**: `Prime Money`
- **Normalized Name**: `prime money`
- **Aggressive Normalized Name**: `prime money`
- **Name Tokens**: `['prime', 'money']`
- **Phonetic**: `PRMMN`
- **Raw Address**: `17560 Ellis Road, Tahlequah, OK`
- **Normalized Address**: `17560 ellis road tahlequah ok`
- **Postal Code Guess**: `17560`
- **City Guess**: `tahlequah`
- **Street Number Guess**: `17560`
- **Country**: `US`

### Sample 3 (ID: S1-377745466)
- **Raw Name**: `B+ Retail Inc`
- **Normalized Name**: `b retail inc`
- **Aggressive Normalized Name**: `b retail`
- **Name Tokens**: `['b', 'retail', 'inc']`
- **Phonetic**: `PRTLNK`
- **Raw Address**: `1712 Montebello Avenue, Phoenix, AZ`
- **Normalized Address**: `1712 montebello avenue phoenix az`
- **Postal Code Guess**: ``
- **City Guess**: `phoenix`
- **Street Number Guess**: `1712`
- **Country**: `US`
