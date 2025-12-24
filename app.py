# app.py (project root)

from pathlib import Path
import streamlit as st
import torch
from PIL import Image
import io

from src.utils.utils import device_auto
from src.train.train import train_and_save
from src.infer.infer import load_model, predict_image_bytes
from src.models.CNN import CNN


def main():
    st.set_page_config(page_title="CNN Train & Infer", layout="centered")
    st.title("CNN 训练 / 推理 Web 界面")

    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "train"
    ckpt_path = root / "checkpoints" / "cnn.pt"

    dev = device_auto()
    st.write(f"**Device:** `{dev}`")

    # =========================
    # Train
    # =========================
    st.divider()
    st.header("训练")

    per_class = st.number_input("每类随机取样张数", min_value=1, value=1000, step=50)
    epochs = st.number_input("训练轮数", min_value=1, value=5, step=1)
    lr = st.number_input("学习率", min_value=1e-6, value=1e-3, step=1e-4, format="%.6f")

    if st.button("开始训练"):
        if not data_dir.exists():
            st.error(f"找不到数据目录：{data_dir}")
            st.stop()

        prog = st.progress(0.0)
        log = st.empty()

        def cb(ep, eps, acc):
            log.write(f"epoch {ep}/{eps} | val_acc={acc:.4f}")
            prog.progress(ep / eps)

        model, saved_path, _ = train_and_save(
            project_root=root,
            per_class=int(per_class),
            epochs=int(epochs),
            lr=float(lr),
            device=dev,
            progress_cb=cb,
        )

        st.session_state["model_state"] = model.state_dict()
        st.success(f"训练完成，已保存：{saved_path}")

    # =========================
    # Infer
    # =========================
    st.divider()
    st.header("上传图片推理")

    up = st.file_uploader("上传 png/jpg", type=["png", "jpg", "jpeg"])
    if up is not None:
        img_bytes = up.getvalue()  

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(img, caption="上传的图片", width=220)

        if "model_state" in st.session_state:
            model = CNN().to(dev)
            model.load_state_dict(st.session_state["model_state"])
            model.eval()
        else:
            if not ckpt_path.exists():
                st.warning(f"没有找到权重文件：{ckpt_path}，请先训练。")
                st.stop()
            model, _ = load_model(ckpt_path, device=dev)

        pred = predict_image_bytes(img_bytes, model, dev)
        st.write(f"**预测结果：** `{pred}`")


if __name__ == "__main__":
    main()