import streamlit as st
import torch
import torch.nn as nn
import re
from transformers import MBartForConditionalGeneration
from indobenchmark import IndoNLGTokenizer

# 1. KONFIGURASI HALAMAN STREAMLIT
st.set_page_config(page_title="Sistem AES IndoBART", page_icon="📝", layout="wide")

# 2. DEFINISI ULANG ARSITEKTUR MODEL (Wajib sama persis dengan Colab)
class IndoBARTMultiTask(nn.Module):
    def __init__(self, model_name="indobenchmark/indobart-v2"):
        super(IndoBARTMultiTask, self).__init__()
        self.indobart = MBartForConditionalGeneration.from_pretrained(model_name)
        hidden_size = self.indobart.config.d_model
        
        self.language_modeling_head = self.indobart.lm_head
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.indobart(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, output_hidden_states=True
        )
        pooled_output = torch.mean(outputs.encoder_last_hidden_state, dim=1)
        score_preds_raw = self.regression_head(pooled_output)
        score_preds = (score_preds_raw * 3.0) + 1.0 
        return score_preds, outputs.logits, outputs.loss

# 3. FUNGSI CACHE UNTUK MEMUAT MODEL (Agar web tidak berat saat di-refresh)
@st.cache_resource
def load_model():
    model_name = "indobenchmark/indobart-v2"
    tokenizer = IndoNLGTokenizer.from_pretrained(model_name)
    
    model = IndoBARTMultiTask(model_name)
    # Load bobot yang sudah dilatih (Pastikan nama file sesuai dengan yang Anda unduh)
    model.load_state_dict(torch.load("model_indobart_aes_final.pt", map_location=torch.device('cpu')))
    model.eval()
    
    return tokenizer, model

tokenizer, model = load_model()

def bersihkan_teks(teks):
    teks = str(teks)
    teks = re.sub(r'[^A-Za-z0-9\s.,?!\-\n"\']', '', teks)
    teks = re.sub(r'[ \t]+', ' ', teks).strip()
    return teks

# 4. DESAIN ANTARMUKA WEB
st.title("📝 Penilaian Esai Otomatis (AES) berbasis AI")
st.markdown("Sistem ini menggunakan arsitektur **Multi-Task IndoBART** untuk mengevaluasi teks argumentasi dan teks berita tingkat SMA.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Data")
    soal = st.text_area("Masukkan Teks Soal:", height=150, placeholder="Ketik soal di sini...")
    jawaban = st.text_area("Masukkan Jawaban Esai Siswa:", height=300, placeholder="Ketik esai siswa di sini...")
    tombol_nilai = st.button("🚀 Nilai Esai Sekarang", use_container_width=True)

with col2:
    st.subheader("Hasil Evaluasi AI")
    if tombol_nilai:
        if not soal or not jawaban:
            st.warning("Mohon isi teks soal dan jawaban terlebih dahulu!")
        else:
            with st.spinner('AI sedang membaca dan mengevaluasi esai...'):
                # Preprocessing
                teks_input = f"Soal: {bersihkan_teks(soal)} Jawaban: {bersihkan_teks(jawaban)}"
                input_tokenized = tokenizer(teks_input, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
                
                input_ids = input_tokenized["input_ids"]
                attention_mask = input_tokenized["attention_mask"]
                
                # Inferensi Multi-Task
                with torch.no_grad():
                    # 1. Tebak Skor
                    score_preds, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    skor_makro1 = round(score_preds[0][0].item()) # Dibulatkan sesuai standar
                    skor_makro2 = round(score_preds[0][1].item())
                    
                    # 2. Bangkitkan Alasan (Rationale)
                    generated_ids = model.indobart.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_length=256, num_beams=4, early_stopping=True
                    )
                    alasan = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Menampilkan Hasil
                sc1, sc2 = st.columns(2)
                sc1.metric(label="Skor Makro 1 (Substansi)", value=f"{skor_makro1} / 4")
                sc2.metric(label="Skor Makro 2 (Kaidah PUEBI)", value=f"{skor_makro2} / 4")
                
                st.success("Teks Alasan (Rationale):")
                st.write(alasan)
    else:
        st.info("Menunggu input dari pengguna... Klik tombol di samping untuk memulai evaluasi.")