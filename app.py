import streamlit as st
import numpy as np, cv2, pickle
from PIL import Image

pca = pickle.load(open('pca.pkl', 'rb'))
region_pcas = pickle.load(open('region_pcas.pkl', 'rb'))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

regions = {'Forehead': (4, 20, 12, 52), 'Left eye': (20, 31, 8, 30), 'Right eye': (20, 31, 34, 56),
           'Nose': (28, 40, 22, 42), 'Mouth': (38, 50, 16, 48), 'Chin': (50, 62, 16, 48)}

def prep(file):
    img = np.array(Image.open(file).convert('L'))
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) > 0:                                   # crop largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        img = img[y:y + h, x:x + w]
    img = cv2.resize(img, (64, 64))
    return cv2.equalizeHist(img)

def patch(imgs, box):
    r1, r2, c1, c2 = box
    return np.array([cv2.GaussianBlur(im, (5, 5), 0)[r1:r2, c1:c2].reshape(-1) for im in imgs]).astype(float)

def compare_parts(f, m, c):
    big = cv2.cvtColor(cv2.resize(c, (256, 256)), cv2.COLOR_GRAY2RGB)
    overlay = big.copy()
    result = {}
    for name, box in regions.items():
        r1, r2, c1, c2 = box
        p = region_pcas[name]
        oc, of, om = [p.transform(patch([x], box)) for x in (c, f, m)]
        dF, dM = np.linalg.norm(oc - of), np.linalg.norm(oc - om)
        result[name] = 'Father' if dF < dM else 'Mother'
        color = (0, 120, 255) if dF < dM else (255, 80, 160)   # RGB: blue = father, pink = mother
        cv2.rectangle(overlay, (c1 * 4, r1 * 4), (c2 * 4, r2 * 4), color, -1)
    return cv2.addWeighted(overlay, 0.35, big, 0.65, 0), result

st.title("Which parent does the child look like?")

c1, c2, c3 = st.columns(3)
f_file = c1.file_uploader("Father", type=['jpg', 'jpeg', 'png'])
m_file = c2.file_uploader("Mother", type=['jpg', 'jpeg', 'png'])
c_file = c3.file_uploader("Child", type=['jpg', 'jpeg', 'png'])

if f_file and m_file and c_file:
    f, m, c = prep(f_file), prep(m_file), prep(c_file)
    oF, oM, oC = [pca.transform(x.reshape(1, -1)) for x in (f, m, c)]

    dF = np.linalg.norm(oC - oF)
    dM = np.linalg.norm(oC - oM)
    pF = (1 / dF) / (1 / dF + 1 / dM)       # inverse distance share

    st.image([f, m, c, (f / 2 + m / 2).astype(np.uint8)],
             caption=['Father', 'Mother', 'Child', "Parents' average"], width=150)

    if dF < dM:
        st.success(f"Looks more like the FATHER ({pF:.0%} vs {1 - pF:.0%})")
    else:
        st.success(f"Looks more like the MOTHER ({1 - pF:.0%} vs {pF:.0%})")

    st.caption("Demo only: PCA on pixels matches true parents about 59% of the time (chance is 50%).")

    img, parts = compare_parts(f, m, c)
    st.image(img, caption="Blue = like father, Pink = like mother", width=300)
    for name, who in parts.items():
        st.write(f"{name}: {who}")