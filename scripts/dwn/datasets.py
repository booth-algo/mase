"""Dataset loading functions for DWN training.

All dataset loaders return tensors ready for DWN training. Heavy dependencies
(torchvision, torchaudio, sklearn, pandas) are imported lazily inside each
loader so the module can be imported without them installed.

Data directory
--------------
By default, datasets are cached under ``~/.cache/dwn``.  Override by setting
the ``DWN_DATA_DIR`` environment variable::

    export DWN_DATA_DIR=/data/datasets
    python run_dwn_training.py --dataset mnist
"""

import os

import torch

# Configurable data root -- honours DWN_DATA_DIR env var, falls back to ~/.cache/dwn
_DATA_DIR = os.environ.get("DWN_DATA_DIR", os.path.expanduser("~/.cache/dwn"))

TABULAR_DATASETS = [
    "phoneme", "skin-seg", "higgs", "australian", "nomao",
    "segment", "miniboone", "christine", "jasmine", "sylvine", "blood",
]

# Datasets with OpenML aliases (user-facing name -> OpenML dataset name)
DATASET_ALIASES = {
    "jsc": "hls4ml_lhc_jets_hlf",  # Jet Substructure Classification (7 features, 5 classes)
}


def load_data(args):
    """Load dataset. Returns (X_train, y_train, X_test, y_test, input_features, num_classes, X_train_base).

    X_train_base is the unaugmented X_train used for thermometer fitting when --augment-refit
    is set; otherwise it is None.
    """
    dataset = args.dataset
    # Backward compat
    if args.real_mnist:
        dataset = "mnist"

    if dataset in ("mnist", "fashion_mnist", "cifar10"):
        return _load_vision(dataset, args)
    elif dataset == "nid":
        result = _load_nid(args)
        return result + (None,)
    elif dataset == "kws":
        result = _load_kws(args)
        return result + (None,)
    elif dataset == "toyadmos":
        result = _load_toyadmos(args)
        return result + (None,)
    elif dataset in TABULAR_DATASETS or dataset in DATASET_ALIASES:
        openml_name = DATASET_ALIASES.get(dataset, dataset)
        result = _load_tabular(openml_name, args)
        return result + (None,)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. Choose from: mnist, fashion_mnist, cifar10, "
            f"jsc, nid, kws, toyadmos, {', '.join(TABULAR_DATASETS)}"
        )


def _load_vision(dataset_name, args):
    import numpy as np

    if dataset_name in ("mnist", "fashion_mnist"):
        return _load_mnist_openml(dataset_name, args)

    # CIFAR-10: still uses torchvision
    try:
        from torchvision import datasets as tvdatasets, transforms
    except Exception as e:
        raise RuntimeError(
            f"torchvision required for cifar10 but import failed: {e}"
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    num_features = 3072
    cls = tvdatasets.CIFAR10
    cache = os.path.join(_DATA_DIR, dataset_name)
    if getattr(args, "augment", False):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    else:
        train_transform = transform
    train_ds = cls(cache, train=True,  download=True, transform=train_transform)
    test_ds  = cls(cache, train=False, download=True, transform=transform)
    X_train = torch.stack([x for x, _ in train_ds])
    y_train = torch.tensor([y for _, y in train_ds])
    X_test  = torch.stack([x for x, _ in test_ds])
    y_test  = torch.tensor([y for _, y in test_ds])

    # When --augment-refit is set, also load unaugmented X_train for thermometer fitting
    X_train_base = None
    if getattr(args, "augment", False) and getattr(args, "augment_refit", False):
        print("augment-refit: loading unaugmented X_train for thermometer fitting...")
        base_ds = cls(cache, train=True, download=False, transform=transform)
        X_train_base = torch.stack([x for x, _ in base_ds])

    print(f"{dataset_name}: {len(X_train)} train, {len(X_test)} test, features={num_features}, classes=10")
    return X_train, y_train, X_test, y_test, num_features, 10, X_train_base


def _load_mnist_openml(dataset_name, args):
    """Load MNIST or FashionMNIST via sklearn fetch_openml (avoids torchvision).

    Caches numpy arrays to <_DATA_DIR>/{mnist,fashion_mnist}/*_features.pt.
    Standard split: first 60,000 = train, last 10,000 = test.
    Pixel values normalised to [0, 1] by dividing by 255.
    """
    import numpy as np
    from sklearn.datasets import fetch_openml

    if dataset_name == "mnist":
        openml_name = "mnist_784"
        cache_dir = os.path.join(_DATA_DIR, "mnist")
        cache_file = os.path.join(cache_dir, "mnist_features.pt")
    else:
        openml_name = "Fashion-MNIST"
        cache_dir = os.path.join(_DATA_DIR, "fashion_mnist")
        cache_file = os.path.join(cache_dir, "fashion_mnist_features.pt")

    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading {dataset_name} features from cache: {cache_file}")
        cached = torch.load(cache_file, map_location="cpu", weights_only=True)
        X_all, y_all = cached["X"], cached["y"]
    else:
        print(f"Fetching {openml_name} from OpenML (one-time download)...")
        data = fetch_openml(openml_name, version=1, as_frame=False, parser="auto")
        X_all = data.data.astype(np.float32) / 255.0
        # Labels may be strings ("0".."9"); convert to int
        y_raw = data.target
        if y_raw.dtype.kind in ("U", "S", "O"):
            y_all = y_raw.astype(np.int64)
        else:
            y_all = y_raw.astype(np.int64)
        torch.save({"X": torch.tensor(X_all), "y": torch.tensor(y_all, dtype=torch.long)}, cache_file)
        print(f"Cached to {cache_file}")
        X_all = torch.tensor(X_all)
        y_all = torch.tensor(y_all, dtype=torch.long)

    if not isinstance(X_all, torch.Tensor):
        X_all = torch.tensor(X_all)
    if not isinstance(y_all, torch.Tensor):
        y_all = torch.tensor(y_all, dtype=torch.long)

    # Standard MNIST split: first 60k train, last 10k test
    X_train, y_train = X_all[:60000], y_all[:60000]
    X_test,  y_test  = X_all[60000:], y_all[60000:]

    print(f"{dataset_name}: {len(X_train)} train, {len(X_test)} test, features=784, classes=10")
    return X_train, y_train, X_test, y_test, 784, 10, None


def _load_tabular(dataset_name, args):
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
    except ImportError:
        raise ImportError("scikit-learn is required for tabular datasets: pip install scikit-learn")

    print(f"Fetching {dataset_name} from OpenML...")
    data = fetch_openml(name=dataset_name, version=1, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y_raw = data.target

    # Encode labels as integers 0..K-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    num_classes = len(le.classes_)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)

    input_features = X_train.shape[1]
    print(f"{dataset_name}: {len(X_train)} train, {len(X_test)} test, "
          f"features={input_features}, classes={num_classes}")
    return X_train, y_train, X_test, y_test, input_features, num_classes


def _load_nid(args):
    """Load NSL-KDD Network Intrusion Detection dataset.

    Downloads KDDTrain+.txt and KDDTest+.txt from the ISCX repository
    if not already cached. 41 features, 5 classes (normal + 4 attack types).
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import urllib.request
    import pandas as pd

    cache_dir = os.path.join(_DATA_DIR, "nsl-kdd")
    os.makedirs(cache_dir, exist_ok=True)

    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    test_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    train_path = os.path.join(cache_dir, "KDDTrain+.txt")
    test_path  = os.path.join(cache_dir, "KDDTest+.txt")

    for url, path in [(train_url, train_path), (test_url, test_path)]:
        if not os.path.exists(path):
            print(f"Downloading NSL-KDD from {url}...")
            urllib.request.urlretrieve(url, path)

    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"
    ]

    df_train = pd.read_csv(train_path, header=None, names=col_names)
    df_test  = pd.read_csv(test_path,  header=None, names=col_names)

    df_train = df_train.drop("difficulty", axis=1)
    df_test  = df_test.drop("difficulty", axis=1)

    cat_cols = ["protocol_type", "service", "flag"]
    df_all = pd.concat([df_train, df_test], axis=0)
    df_all = pd.get_dummies(df_all, columns=cat_cols)
    df_train2 = df_all.iloc[:len(df_train)]
    df_test2  = df_all.iloc[len(df_train):]

    attack_map = {
        "normal": "normal",
        "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos",
        "smurf": "dos", "teardrop": "dos", "mailbomb": "dos", "apache2": "dos",
        "processtable": "dos", "udpstorm": "dos",
        "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
        "mscan": "probe", "saint": "probe",
        "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
        "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
        "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
        "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
        "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
        "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    }

    y_train_raw = df_train2["class"].map(lambda x: attack_map.get(x, "other"))
    y_test_raw  = df_test2["class"].map(lambda x: attack_map.get(x, "other"))

    X_train_np = df_train2.drop("class", axis=1).values.astype(np.float32)
    X_test_np  = df_test2.drop("class", axis=1).values.astype(np.float32)

    le = LabelEncoder()
    le.fit(pd.concat([y_train_raw, y_test_raw]))
    y_train_np = le.transform(y_train_raw).astype(np.int64)
    y_test_np  = le.transform(y_test_raw).astype(np.int64)

    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)

    num_classes = len(le.classes_)
    input_features = X_train.shape[1]
    print(f"nid (NSL-KDD): {len(X_train)} train, {len(X_test)} test, "
          f"features={input_features}, classes={num_classes} {list(le.classes_)}")
    return X_train, y_train, X_test, y_test, input_features, num_classes


def _load_kws(args):
    """Load Google Speech Commands v2 (MLPerf Tiny KWS subset, 12 classes).

    Features: 51 frames x 10 MFCC coefficients = 510 input features (torchaudio uses center-padding).
    Paper config (Table 14): z=8 thermometer bits, hidden=1608, n=6, 100 epochs.
    Uses soundfile for WAV loading (avoids torchcodec dependency in torchaudio 2.10+).
    Downloads ~2.3GB to <_DATA_DIR>/speech_commands/.
    Caches extracted features to kws_features.pt for fast subsequent loads.
    """
    import numpy as np
    import glob

    cache_dir = os.path.join(_DATA_DIR, "speech_commands")
    os.makedirs(cache_dir, exist_ok=True)
    features_cache = os.path.join(cache_dir, "kws_features.pt")

    # Return cached features if available
    if os.path.exists(features_cache):
        print(f"Loading KWS features from cache: {features_cache}")
        cached = torch.load(features_cache, map_location="cpu", weights_only=True)
        X_train, y_train = cached["X_train"], cached["y_train"]
        X_test,  y_test  = cached["X_test"],  cached["y_test"]
        num_features = X_train.shape[1]
        print(f"kws: {len(X_train)} train, {len(X_test)} test, "
              f"features={num_features}, classes=12")
        return X_train, y_train, X_test, y_test, num_features, 12

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for KWS: pip install soundfile")

    try:
        import torchaudio
        import torchaudio.transforms as T
    except ImportError:
        raise ImportError("torchaudio is required for KWS: pip install torchaudio")

    # Download if needed
    data_dir = os.path.join(cache_dir, "SpeechCommands", "speech_commands_v0.02")
    if not os.path.exists(data_dir):
        print(f"Downloading Google Speech Commands v2 (~2.3GB) to {cache_dir}...")
        import urllib.request, tarfile
        url = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        tar_path = os.path.join(cache_dir, "speech_commands_v0.02.tar.gz")
        os.makedirs(os.path.join(cache_dir, "SpeechCommands", "speech_commands_v0.02"), exist_ok=True)
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as tar:
            dest = os.path.join(cache_dir, "SpeechCommands", "speech_commands_v0.02")
            for member in tar.getmembers():
                if os.path.isabs(member.name) or ".." in member.name:
                    raise RuntimeError(f"Refusing to extract path-traversal entry: {member.name}")
            tar.extractall(dest)

    # MLPerf Tiny KWS spec: 30ms window, 20ms stride, 10 MFCC, 40 mel bins
    SAMPLE_RATE = 16000
    N_MFCC = 10
    N_MELS = 40
    WIN_LEN = int(0.030 * SAMPLE_RATE)   # 480 samples
    HOP_LEN = int(0.020 * SAMPLE_RATE)   # 320 samples

    KEYWORDS = ["yes", "no", "up", "down", "left", "right",
                "on", "off", "stop", "go"]
    LABEL_MAP = {w: i for i, w in enumerate(KEYWORDS)}
    LABEL_MAP["_silence_"] = 10
    # unknown = 11

    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_mels": N_MELS,
            "win_length": WIN_LEN,
            "hop_length": HOP_LEN,
            "n_fft": 1024,
            "f_min": 20.0,
            "f_max": 4000.0,
        },
    )

    # Load split lists
    test_list_path = os.path.join(data_dir, "testing_list.txt")
    val_list_path  = os.path.join(data_dir, "validation_list.txt")
    with open(test_list_path) as f:
        test_files = set(line.strip() for line in f)
    with open(val_list_path) as f:
        val_files = set(line.strip() for line in f)

    def wav_to_features(wav_path):
        """Load WAV with soundfile, extract MFCC -> (510,) float32."""
        data, sr = sf.read(wav_path, dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        tgt = SAMPLE_RATE
        if waveform.shape[-1] < tgt:
            waveform = torch.nn.functional.pad(waveform, (0, tgt - waveform.shape[-1]))
        else:
            waveform = waveform[..., :tgt]
        mfcc = mfcc_transform(waveform)  # (1, 10, 51)
        return mfcc.squeeze(0).transpose(0, 1).reshape(-1).numpy()  # (510,)

    print("Loading Google Speech Commands v2 (MLPerf Tiny KWS)...")
    print("  Extracting MFCC features (this takes ~35min first time; cached after)...")

    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []
    skipped = 0

    all_keywords = KEYWORDS + ["_silence_"]
    other_dirs = [d for d in sorted(os.listdir(data_dir))
                  if os.path.isdir(os.path.join(data_dir, d))
                  and d not in {"_background_noise_"}
                  and d not in all_keywords]

    # Process keyword directories (all files, no cap)
    for kw in KEYWORDS + ["_silence_"]:
        kw_dir = os.path.join(data_dir, kw)
        if not os.path.isdir(kw_dir):
            continue
        label = LABEL_MAP[kw]
        wavs = sorted(glob.glob(os.path.join(kw_dir, "*.wav")))
        for wav_path in wavs:
            rel_path = os.path.relpath(wav_path, data_dir)
            # Skip validation files entirely (not needed for train or test)
            if rel_path in val_files:
                continue
            try:
                feat = wav_to_features(wav_path)
            except (RuntimeError, ValueError, OSError) as e:
                skipped += 1
                continue
            if rel_path in test_files:
                X_test_list.append(feat)
                y_test_list.append(label)
            else:
                X_train_list.append(feat)
                y_train_list.append(label)

    # Process "unknown" class from non-keyword dirs
    unknown_train_count = 0
    for d in other_dirs:
        wavs = sorted(glob.glob(os.path.join(data_dir, d, "*.wav")))
        for wav_path in wavs:
            rel_path = os.path.relpath(wav_path, data_dir)
            if rel_path in val_files:
                continue
            is_test = rel_path in test_files
            if not is_test and unknown_train_count >= 2000:
                continue
            try:
                feat = wav_to_features(wav_path)
            except (RuntimeError, ValueError, OSError):
                skipped += 1
                continue
            if is_test:
                X_test_list.append(feat)
                y_test_list.append(11)
            else:
                X_train_list.append(feat)
                y_train_list.append(11)
                unknown_train_count += 1
        # Stop after train cap AND we've seen enough test samples (at least 400)
        unknown_test_count = sum(1 for y in y_test_list if y == 11)
        if unknown_train_count >= 2000 and unknown_test_count >= 400:
            break

    X_train = torch.tensor(np.stack(X_train_list), dtype=torch.float32)
    y_train = torch.tensor(y_train_list, dtype=torch.long)
    X_test  = torch.tensor(np.stack(X_test_list),  dtype=torch.float32)
    y_test  = torch.tensor(y_test_list,  dtype=torch.long)

    # Save to cache
    print(f"  Saving features to cache: {features_cache}")
    torch.save({"X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test}, features_cache)

    if skipped:
        print(f"  Warning: skipped {skipped} unreadable WAV files")
    num_features = X_train.shape[1]  # 490
    print(f"kws: {len(X_train)} train, {len(X_test)} test, "
          f"features={num_features}, classes=12")
    return X_train, y_train, X_test, y_test, num_features, 12


def _load_toyadmos(args):
    """Load DCASE 2020 Task 2 ToyCar anomaly detection dataset.

    Binary classification: normal (0) vs anomalous (1).
    Features: log-mel spectrogram, 128 mel bins x 5 consecutive frames = 640 features.
    Paper config (Table 14): z=3 therm. bits, hidden=[1800,1800], n=6, 100 epochs.
    Downloads ~1.8GB to <_DATA_DIR>/toyadmos/.

    Zip extracts to ToyCar/train/ (all normal) and ToyCar/test/ (normal + anomaly).
    Label is determined by filename prefix: normal_* -> 0, anomaly_* -> 1.
    """
    import numpy as np
    import urllib.request
    import zipfile
    import glob
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for ToyADMOS: pip install soundfile")

    cache_dir = os.path.join(_DATA_DIR, "toyadmos")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        import torchaudio
        import torchaudio.transforms as T
    except ImportError:
        raise ImportError(
            "torchaudio is required for ToyADMOS: "
            "conda run -n plena2 pip install torchaudio"
        )

    # DCASE 2020 Task 2 ToyCar dev dataset (1.8 GB)
    DEV_ZIP_URL = "https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip"
    dev_zip_path = os.path.join(cache_dir, "dev_data_ToyCar.zip")
    dev_data_dir = os.path.join(cache_dir, "ToyCar")

    if not os.path.exists(dev_data_dir):
        if not os.path.exists(dev_zip_path):
            print(f"Downloading ToyADMOS/car dev set (~1.8GB) to {cache_dir}...")
            urllib.request.urlretrieve(DEV_ZIP_URL, dev_zip_path)
            print("  Download complete.")
        print("  Extracting...")
        with zipfile.ZipFile(dev_zip_path, "r") as zf:
            for name in zf.namelist():
                if os.path.isabs(name) or ".." in name:
                    raise RuntimeError(f"Refusing to extract path-traversal entry: {name}")
            zf.extractall(cache_dir)

    features_cache = os.path.join(cache_dir, "toyadmos_features.pt")
    if os.path.exists(features_cache):
        print(f"Loading cached features from {features_cache}...")
        cached = torch.load(features_cache, map_location="cpu", weights_only=True)
        X_train = cached["X_train"]
        y_train = cached["y_train"]
        X_test  = cached["X_test"]
        y_test  = cached["y_test"]
        num_features = X_train.shape[1]
        print(f"toyadmos: {len(X_train)} train (all normal), {len(X_test)} test "
              f"(normal+anomaly), features={num_features}, classes=2")
        return X_train, y_train, X_test, y_test, num_features, 2

    # MLPerf Tiny anomaly detection spec: 128 mel, 5 frames, 1024 FFT, 512 hop
    SAMPLE_RATE = 16000
    N_MELS = 128
    N_FRAMES = 5
    N_FFT = 1024
    HOP_LEN = 512

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
    )

    def extract_features(wav_path):
        """Extract 128x5 log-mel spectrogram patch -> 640-dim feature vector."""
        data, sr = sf.read(wav_path, dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        # Trim to center segment: only compute mel for frames we actually need
        # This gives ~50x speedup over full 11s audio
        context_samples = (N_FRAMES + 4) * HOP_LEN + N_FFT
        center = waveform.shape[-1] // 2
        start = max(0, center - context_samples // 2)
        end = min(waveform.shape[-1], start + context_samples)
        waveform = waveform[:, start:end]
        mel = mel_transform(waveform)  # (1, 128, ~9 frames)
        log_mel = torch.log(mel + 1e-8).squeeze(0)  # (128, T)
        # Take center N_FRAMES frames
        mid = log_mel.shape[1] // 2
        start_frame = max(0, mid - N_FRAMES // 2)
        patch = log_mel[:, start_frame:start_frame + N_FRAMES]  # (128, 5)
        if patch.shape[1] < N_FRAMES:
            patch = torch.nn.functional.pad(patch, (0, N_FRAMES - patch.shape[1]))
        return patch.transpose(0, 1).reshape(-1).numpy()  # (640,)

    print("Loading ToyADMOS/car dataset (DCASE 2020 Task 2)...")
    skipped_wav = 0

    # Train: all files in ToyCar/train/ are normal (label=0)
    print("  Processing train split (ToyCar/train/, all normal)...")
    train_dir = os.path.join(dev_data_dir, "train")
    X_tr, y_tr = [], []
    for wav in sorted(glob.glob(os.path.join(train_dir, "*.wav"))):
        fname = os.path.basename(wav)
        if fname.startswith("normal_"):
            try:
                X_tr.append(extract_features(wav))
                y_tr.append(0)
            except (RuntimeError, ValueError, OSError):
                skipped_wav += 1

    # Test: files in ToyCar/test/ labeled by filename prefix
    print("  Processing test split (ToyCar/test/, normal + anomaly)...")
    test_dir = os.path.join(dev_data_dir, "test")
    X_te, y_te = [], []
    for wav in sorted(glob.glob(os.path.join(test_dir, "*.wav"))):
        fname = os.path.basename(wav)
        if fname.startswith("normal_"):
            label = 0
        elif fname.startswith("anomaly_"):
            label = 1
        else:
            continue
        try:
            X_te.append(extract_features(wav))
            y_te.append(label)
        except (RuntimeError, ValueError, OSError):
            skipped_wav += 1

    if skipped_wav:
        print(f"  Warning: skipped {skipped_wav} unreadable WAV files")
    X_train = torch.tensor(np.stack(X_tr), dtype=torch.float32)
    y_train = torch.tensor(y_tr, dtype=torch.long)
    X_test  = torch.tensor(np.stack(X_te), dtype=torch.float32)
    y_test  = torch.tensor(y_te, dtype=torch.long)

    # Save to cache
    print(f"  Saving features to cache: {features_cache}")
    torch.save({"X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test}, features_cache)

    num_features = X_train.shape[1]  # 640
    print(f"toyadmos: {len(X_train)} train (all normal), {len(X_test)} test "
          f"(normal+anomaly), features={num_features}, classes=2")
    return X_train, y_train, X_test, y_test, num_features, 2


def _fake_data(args):
    print(f"Using fake random data ({args.n_train} train samples)")
    X_train = torch.randn(args.n_train, 784)
    y_train = torch.randint(0, 10, (args.n_train,))
    X_test  = torch.randn(200, 784)
    y_test  = torch.randint(0, 10, (200,))
    return X_train, y_train, X_test, y_test, 784, 10, None
