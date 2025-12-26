from pathlib import Path
import io
import time

import streamlit as st
from PIL import Image

from src.utils.utils import device_auto
from src.train.train import train_and_save
from src.infer.infer import load_model, predict_image_bytes
from src.models.CNN import CNN
from src.infer.infer_test import evaluate_testset


def main():
    st.set_page_config(page_title="CNN Train & Infer", layout="centered")
    st.title("CNN 训练 / 推理 Web 界面")

    root = Path(__file__).resolve().parent
    data_train_dir = root / "data" / "train"
    data_test_dir = root / "data" / "test"
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
        if not data_train_dir.exists():
            st.error(f"找不到训练数据目录：{data_train_dir}")
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

        # 优先使用 session 中的模型（如果网页里刚训练过）
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

    # =========================
    # Local test mode
    # =========================
    st.divider()
    st.header("本地测试集测试模式")

    enable_test = st.checkbox("启用本地测试集测试模式", value=False)

    if enable_test:
        if not data_test_dir.exists():
            st.error(f"找不到测试数据目录：{data_test_dir}")
            st.stop()
        if not ckpt_path.exists() and "model_state" not in st.session_state:
            st.error(f"找不到权重：{ckpt_path}请先训练。")
            st.stop()

        if "test_result" not in st.session_state:
            with st.spinner("正在测试 5000 张图片，请稍候..."):
                overall_acc, per_class_acc, samples = evaluate_testset(
                    project_root=root,
                    ckpt_path=ckpt_path,
                    batch_size=256,
                    seed=42,
                    device=dev,
                    sample_per_class=10,
                    sample_seed=int(time.time()),  # 样例每次测试都随机
                )
            st.session_state["test_result"] = (overall_acc, per_class_acc, samples)

        if st.button("重新测试"):
            st.session_state.pop("test_result", None)
            st.rerun()

        overall_acc, per_class_acc, samples = st.session_state["test_result"]

        st.subheader("测试结果")
        st.write(f"**总正确率（5000 张）：** `{overall_acc:.4f}`")

        st.write("**各类别正确率：**")
        for k in range(10):
            st.write(f"- 类别 `{k}`：`{per_class_acc.get(k, 0.0):.4f}`")

        st.subheader("样例检查")
        for k in range(10):
            st.markdown(f"### 类别 {k}")
            cols = st.columns(10)
            for j, (img_path, pred) in enumerate(samples.get(k, [])):
                with cols[j]:
                    img = Image.open(img_path).convert("RGB")
                    st.image(img, caption=f"true={k}, pred={pred}", use_container_width=True)


if __name__ == "__main__":
    main()