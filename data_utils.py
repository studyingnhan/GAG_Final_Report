"""
data_utils.py · Utilities cho dự án Gender & Age Classification
---------------------------------------------------------------
- Nếu base_dir chứa sẵn train / val / test  → dùng trực tiếp.
- Nếu không, hàm sẽ tự chia theo val_split, test_split.
- Trả về: train_ds, val_ds, test_ds, class_weights
  (Dataset đã batch + prefetch, class_weights cho 2 đầu ra)
"""

from __future__ import annotations
import os, glob, numpy as np, tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ───── hằng số & class list ────────────────────────────────────────────
AUTOTUNE        = tf.data.AUTOTUNE
GENDER_CLASSES  = ["male", "female"]
AGE_CLASSES     = ["child", "teen", "adult", "elderly"]

# ───── augmentation layer ──────────────────────────────────────────────
def build_augmentation(img_size: int = 224) -> tf.keras.Sequential:
    """Layer augmentation áp dụng on-the-fly ở pipeline train."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )

# ───── helpers ─────────────────────────────────────────────────────────
def _decode(path: tf.Tensor, img_size: int) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32) / 255.0


def _encode(folder: tf.Tensor):
    """Trả về one-hot giới tính & tuổi từ tên thư mục (b'male_child')."""
    folder = tf.strings.lower(folder)
    g_idx = tf.where(tf.strings.regex_full_match(folder, b"male.*"), 0, 1)

    age_idx = tf.argmax(
        tf.stack(
            [
                tf.strings.regex_full_match(folder, b".*child"),
                tf.strings.regex_full_match(folder, b".*teen"),
                tf.strings.regex_full_match(folder, b".*adult"),
                tf.strings.regex_full_match(folder, b".*elderly"),
            ]
        ),
        axis=0,
    )
    return tf.one_hot(g_idx, 2), tf.one_hot(age_idx, 4)


def _load_item(path: tf.Tensor, img_size: int):
    img = _decode(path, img_size)
    folder = tf.strings.split(path, os.sep.encode())[-2]  # b'male_child'
    g_oh, a_oh = _encode(folder)
    return img, {"gender_output": g_oh, "age_output": a_oh}


def _make_ds(
    files: list[str],
    img_size: int,
    batch: int,
    shuffle: bool,
    seed: int,
    augment: bool,
    aug_layer: tf.keras.layers.Layer | None,
):
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(files), 1000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(lambda p: _load_item(p, img_size), num_parallel_calls=AUTOTUNE)

    if augment and aug_layer is not None:
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    return ds.batch(batch).prefetch(AUTOTUNE)


def _list_files(folder: str) -> list[str]:
    """Duyệt đệ quy mọi ảnh .jpg|.jpeg|.png dưới folder."""
    exts = ("jpg", "jpeg", "png")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    return files


def _compute_class_weights(files: list[str]) -> dict:
    g_idx, a_idx = [], []
    for p in files:
        folder = os.path.basename(os.path.dirname(p)).lower()  # male_child
        g_idx.append(0 if folder.startswith("male") else 1)
        a_idx.append(AGE_CLASSES.index(folder.split("_")[-1]))
    cw_gender = compute_class_weight("balanced", classes=np.unique(g_idx), y=g_idx)
    cw_age = compute_class_weight("balanced", classes=np.unique(a_idx), y=a_idx)
    return {
        "gender_output": {i: float(w) for i, w in enumerate(cw_gender)},
        "age_output": {i: float(w) for i, w in enumerate(cw_age)},
    }

# ───── public API ──────────────────────────────────────────────────────
def get_datasets(
    base_dir: str,
    img_size: int = 224,
    batch: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    augmentation: bool = True,
):
    """
    Trả về: train_ds, val_ds, test_ds, class_weights
    - Nếu base_dir có sẵn /train /val /test thì dùng nguyên.
    - Ngược lại, tự chia theo val_split & test_split.
    """
    base_dir = os.path.abspath(base_dir)
    has_split = all(os.path.exists(os.path.join(base_dir, d)) for d in ("train", "val", "test"))

    if has_split:
        train_files = _list_files(os.path.join(base_dir, "train"))
        val_files = _list_files(os.path.join(base_dir, "val"))
        test_files = _list_files(os.path.join(base_dir, "test"))
    else:
        all_files = _list_files(base_dir)
        if not all_files:
            raise RuntimeError("❗ Không tìm thấy ảnh trong thư mục đã chỉ định.")
        all_files = tf.random.shuffle(all_files, seed=seed).numpy().tolist()
        n = len(all_files)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test_files = all_files[:n_test]
        val_files = all_files[n_test : n_test + n_val]
        train_files = all_files[n_test + n_val :]

    if not train_files:
        raise RuntimeError("❗ Thư mục train rỗng – kiểm tra cấu trúc dataset.")

    aug_layer = build_augmentation(img_size) if augmentation else None
    train_ds = _make_ds(train_files, img_size, batch, True, seed, augmentation, aug_layer)
    val_ds = _make_ds(val_files, img_size, batch, False, seed, False, None)
    test_ds = _make_ds(test_files, img_size, batch, False, seed, False, None)

    class_weights = _compute_class_weights(train_files)
    return train_ds, val_ds, test_ds, class_weights
