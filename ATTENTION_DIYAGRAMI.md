# Büyük Dil Modelleri İçin Attention (Dikkat) Mekanizması Akış Şeması

Bu şema, ham metnin adım adım işlenerek `Multi-Head Attention` bloğundan nasıl geçtiğini gösteren genel bir röntgendir. Transformer mimarisinin kalbini oluşturan bu sürecin detaylı öğretimi [Bölüm 3 özetinde](ch03/01_main-chapter-code/BOLUM3_OZETI.md) yer almaktadır.

```mermaid
""graph TD
    %% Renk ve Stil Tanımlamaları
    classDef inputData fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
	    classDef processing fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef matrix fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef attention fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef final fill:#ede7f6,stroke:#512da8,stroke-width:2px,color:#000Altyazı M.K. 

    %% 1. Veri Hazırlama (Bölüm 2)
    subgraph Girdi_Hazirlama ["1. Girdi ve Gömme (Input & Embeddings)"]
        A["Ham Metin (Text)"]:::inputData --> B["BPE Tokenizer"]:::processing
        B --> C["Token ID'leri (Sayılar)"]:::inputData
        C --> D["Token Embedding (Anlam)"]:::processing
        E["Sıra Numaraları"]:::inputData --> F["Positional Embedding (Konum)"]:::processing
        D --> G{"+ Toplama"}:::matrix
        F --> G
        G --> H["Girdi Matrisi (Input - X)"]:::matrix
    end

    %% 2. Ağırlıklar ile Projeksiyon (Bölüm 3.4)
    subgraph Projeksiyon ["2. Çarpan Matrisleri (Linear Projections)"]
        H -->|X @ W_q| Q_Head1["Sorgu (Query - Q)"]:::matrix
        H -->|X @ W_k| K_Head1["Anahtar (Key - K)"]:::matrix
        H -->|X @ W_v| V_Head1["Değer (Value - V)"]:::matrix
    end

    %% 3. Tek Başına Self-Attention ve Maskeleme (Bölüm 3.4 & 3.5)
    subgraph Self_Attention_Kafasi ["3. Self-Attention (Tek Bir Kafa / Head 1)"]
        Q_Head1 -->|Q @ K.T| I["Dikkat Skorları (Attention Scores)"]:::attention
        K_Head1 --> I
        I -->|Karekök d_k'ya böl| J["Ölçekleme (Scaling)"]:::processing
        J --> K["Causal Masking<br>(Geleceği -Sonsuz yap)"]:::processing
        K --> L["Softmax (Skorları %'ye çevir)"]:::processing
        L --> M["Dikkat Ağırlıkları (Attn Weights)"]:::attention
  
        M -->|Weights @ V| N["Dropout (Rastgele Unutma)"]:::processing
        V_Head1 --> N
        N --> O["Bağlam Vektörü (Context Vector - Z1)"]:::final
    end

    %% 4. Çoklu Kafa (Multi-Head) Mimarisi (Bölüm 3.6)
    subgraph Multi_Head ["4. Cümlenin Çoklu Boyutta İncelenmesi (Multi-Head Attention)"]
        H -.->|Head 2| Head2["Bağlam Vektörü 2 (Z2)"]:::final
        H -.->|...| HeadN["Bağlam Vektörü N (Zn)"]:::final
  
        O --> P{"Birleştirme (Concatenate)"}:::matrix
        Head2 --> P
        HeadN --> P
  
        P --> Q["Birleşik Matris"]:::matrix
        Q -->|W_out ile Çarp| R["Nihai Çıktı (Final Output)"]:::final
    end

    %% Genel Akış
    Girdi_Hazirlama --> Projeksiyon
    Projeksiyon --> Self_Attention_Kafasi
    Self_Attention_Kafasi --> Multi_Head
```
