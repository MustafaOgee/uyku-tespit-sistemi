import cv2
import mediapipe as mp
import pygame
import math
import matplotlib
matplotlib.use('Agg') # Arka plan çökmelerini engeller

# --- 1. SES SİSTEMİNİ BAŞLAT ---
pygame.mixer.init()
alarm_sesi = "alarm.mp3"

# --- 2. MEDIAPIPE YÜZ TANIMA AYARLARI ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

SOL_GOZ = [362, 385, 387, 263, 373, 380]
SAG_GOZ = [33, 160, 158, 133, 153, 144]

def mesafe_hesapla(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def ear_hesapla(goz_noktalari, yuz_isaretleri):
    p1 = yuz_isaretleri.landmark[goz_noktalari[0]]
    p2 = yuz_isaretleri.landmark[goz_noktalari[1]]
    p3 = yuz_isaretleri.landmark[goz_noktalari[2]]
    p4 = yuz_isaretleri.landmark[goz_noktalari[3]]
    p5 = yuz_isaretleri.landmark[goz_noktalari[4]]
    p6 = yuz_isaretleri.landmark[goz_noktalari[5]]

    dikey_1 = mesafe_hesapla(p2, p6)
    dikey_2 = mesafe_hesapla(p3, p5)
    yatay = mesafe_hesapla(p1, p4)

    if yatay == 0:
        return 0
    return (dikey_1 + dikey_2) / (2.0 * yatay)

# --- 3. UYKU ALGILAMA AYARLARI ---
EAR_ESIK_DEGERI = 0.22  
UYKU_SURESI_ESIGI = 15  
kapali_kare_sayaci = 0
alarm_caliyor_mu = False

# Kamerayı aç
kamera = cv2.VideoCapture(0)

print("Sistem çalışıyor... Çıkmak için ekrandayken 'q' tuşuna bas.")

while True:
    ret, frame = kamera.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sonuc = face_mesh.process(rgb_frame)

    if sonuc.multi_face_landmarks:
        for yuz_isaretleri in sonuc.multi_face_landmarks:
            sol_ear = ear_hesapla(SOL_GOZ, yuz_isaretleri)
            sag_ear = ear_hesapla(SAG_GOZ, yuz_isaretleri)
            ortalama_ear = (sol_ear + sag_ear) / 2.0

            # Ekrana göz açıklık oranını yaz
            cv2.putText(frame, f"Goz Acikligi: {ortalama_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if ortalama_ear < EAR_ESIK_DEGERI:
                kapali_kare_sayaci += 1
                if kapali_kare_sayaci >= UYKU_SURESI_ESIGI:
                    cv2.putText(frame, "UYAN YEGEN!", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    
                    if not alarm_caliyor_mu:
                        try:
                            pygame.mixer.music.load(alarm_sesi)
                            pygame.mixer.music.play(-1) 
                            alarm_caliyor_mu = True
                        except Exception as e:
                            print(f"Hata: Ses dosyasi bulunamadi! ({e})")
            else:
                kapali_kare_sayaci = 0
                if alarm_caliyor_mu:
                    pygame.mixer.music.stop()
                    alarm_caliyor_mu = False

    cv2.imshow("Uyku Takip Sistemi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()