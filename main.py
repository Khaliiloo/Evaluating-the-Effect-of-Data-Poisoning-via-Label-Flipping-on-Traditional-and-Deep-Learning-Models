# -*- coding: utf-8 -*-
"""
OPTIMIZED Unified Experiment Script for Arabic Offensive Language Detection
- Reduced CV repeats and folds for faster execution
- Optimized BERT training (1 epoch, smaller batch size)
- Selective noise levels
- Early stopping for BERT
- Parallel processing where possible

Runtime: ~30-45 minutes on GPU (vs hours before)

FIXED VERSION - Corrected data preprocessing flow
FULL ERROR ANALYSIS - Comprehensive error reporting and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    average_precision_score,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import wilcoxon
from collections import defaultdict, Counter
import os
import warnings
import re
from nltk.tokenize import word_tokenize
import nltk
import time

import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

warnings.filterwarnings('ignore')

# Ensure NLTK punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ===== OPTIMIZED CONFIGURATION =====
MASTER_SEED = 42
np.random.seed(MASTER_SEED)

EXPERIMENT_MODE = 'cv'  # 'cv' or 'train_test'

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

if EXPERIMENT_MODE == 'train_test':
    N_REPEATS = 5
    results_dir = 'results_revised'
elif EXPERIMENT_MODE == 'cv':
    N_REPEATS = 5
    N_SPLITS = 5
    results_dir = 'results_revised_cv'
else:
    raise ValueError("EXPERIMENT_MODE must be 'cv' or 'train_test'")

ENABLE_HYPERPARAM_SEARCH = False

# Create directories
for d in [results_dir,
          os.path.join(results_dir, 'error_analysis'),
          os.path.join(results_dir, 'error_analysis', 'examples'),
          os.path.join(results_dir, 'error_analysis', 'plots'),
          os.path.join(results_dir, 'error_analysis', 'per_model'),
          os.path.join(results_dir, 'confusion_matrices')]:
    os.makedirs(d, exist_ok=True)

error_dir = os.path.join(results_dir, 'error_analysis')

# ===== BERT SETUP =====
try:
    BERT_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"✓ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        print("⚠ CUDA not available. Using CPU.")
except ImportError:
    BERT_AVAILABLE = False
    DEVICE = None
    print("Transformers not available. Using SimplifiedBERTModel.")


# ===== TOKENIZER =====
class ArabicTokenizer:
    """Arabic tokenizer with normalization."""

    def __init__(self, min_token_length=2):
        self.min_token_length = min_token_length

    def __call__(self, text):
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        text = re.sub(r'ـ', '', text)
        text = re.sub(r'[إأآ]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        return [t for t in tokens if len(t) >= self.min_token_length]


# ===== SIMPLIFIED BERT (FALLBACK) =====
class SimplifiedBERTModel:
    def __init__(self):
        self.name = "BERT-like"
        self.offensive_keywords = []
        self.word_freq = {'offensive': {}, 'non_offensive': {}}

    def fit(self, X, y):
        for text, label in zip(X, y):
            words = text.split()
            for word in words:
                key = 'offensive' if label == 1 else 'non_offensive'
                self.word_freq[key][word] = self.word_freq[key].get(word, 0) + 1
        self.offensive_keywords = [
            word for word in self.word_freq['offensive']
            if (self.word_freq['offensive'][word] >
                self.word_freq['non_offensive'].get(word, 0) * 3 and
                self.word_freq['offensive'][word] > 5)
        ]
        if not self.offensive_keywords:
            self.offensive_keywords = ['مسيء', 'سيء', 'غبي']

    def predict(self, X):
        return np.array([
            1 if any(kw in text for kw in self.offensive_keywords) else 0
            for text in X
        ])

    def predict_proba(self, X):
        preds = self.predict(X)
        probas = np.zeros((len(X), 2))
        for i, pred in enumerate(preds):
            probas[i, pred] = 0.9
            probas[i, 1 - pred] = 0.1
        return probas


# ===== OPTIMIZED BERT MODEL =====
class ArabicBERTModel:
    def __init__(self, model_name='asafaya/bert-base-arabic',
                 max_length=64, batch_size=32, epochs=5, learning_rate=3e-5):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.name = "Arabic BERT"
        if not BERT_AVAILABLE:
            raise RuntimeError("Transformers required for ArabicBERTModel")
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _prepare_data(self, X, y=None):
        encodings = self.tokenizer(
            list(X), truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        if y is not None:
            labels = torch.tensor(list(y), dtype=torch.long)
            dataset = TensorDataset(encodings['input_ids'],
                                    encodings['attention_mask'], labels)
        else:
            dataset = TensorDataset(encodings['input_ids'],
                                    encodings['attention_mask'])
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=(y is not None), num_workers=0)

    def fit(self, X, y):
        start_time = time.time()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2)
        self.model.to(self.device)
        train_loader = self._prepare_data(X, y)
        optimizer = AdamW(self.model.parameters(),
                          lr=self.learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(100, int(0.1 * total_steps)),
            num_training_steps=total_steps)
        self.model.train()
        accumulation_steps = 2
        for epoch in range(self.epochs):
            total_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps
                loss.backward()
                total_loss += outputs.loss.item()
                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(train_loader):.4f} ({elapsed:.1f}s)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def predict(self, X):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in self._prepare_data(X, y=None):
                outputs = self.model(input_ids=batch[0].to(self.device),
                                     attention_mask=batch[1].to(self.device))
                predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        return np.array(predictions)

    def predict_proba(self, X):
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for batch in self._prepare_data(X, y=None):
                outputs = self.model(input_ids=batch[0].to(self.device),
                                     attention_mask=batch[1].to(self.device))
                probabilities.extend(torch.softmax(outputs.logits, dim=1).cpu().numpy())
        return np.array(probabilities)


# ===== DATA LOADING =====
def load_and_preprocess_data():
    try:
        data = pd.read_csv('ardata.csv')
        print(f"Dataset loaded with {len(data)} rows")
        if 'platform' not in data.columns and 'source' not in data.columns:
            data['platform'] = np.random.choice(['Twitter', 'Facebook', 'YouTube'], len(data))
    except FileNotFoundError:
        print("Creating sample data...")
        comments = ["هذا تعليق عادي وليس مسيء", "هذا تعليق مسيء جدا",
                    "أنا أحب هذا المحتوى", "أنت شخص سيء"] * 125
        labels = ['Not Offensive', 'Offensive'] * 250
        platforms = np.random.choice(['Twitter', 'Facebook', 'YouTube'], 500)
        data = pd.DataFrame({'Comment': comments, 'Majority_Label': labels, 'platform': platforms})

    data['Comment'] = data['Comment'].astype(str)
    data['label'] = data['Majority_Label'].apply(lambda x: 1 if x == 'Offensive' else 0)
    if 'platform' not in data.columns:
        data['platform'] = 'Unknown'
    print(f"Class distribution: {data['label'].value_counts().to_dict()}")
    return data[['Comment', 'label', 'platform']]


# ===== LABEL NOISE =====
def introduce_label_noise(data, noise_level=0.1, seed=None, track_flips=True):
    if noise_level == 0.0:
        return data.copy(), None
    data = data.copy()
    if seed is not None:
        np.random.seed(seed)
    n_flip = int(len(data) * noise_level)
    flip_idx = np.random.choice(data.index, size=n_flip, replace=False)
    flip_stats = None
    if track_flips:
        original_labels = data.loc[flip_idx, 'label'].values
        flip_1_to_0 = np.sum(original_labels == 1)
        flip_0_to_1 = np.sum(original_labels == 0)
        flip_stats = {
            'n_flipped': n_flip,
            'flip_1_to_0': flip_1_to_0,
            'flip_0_to_1': flip_0_to_1,
            'prob_1_to_0': flip_1_to_0 / n_flip if n_flip > 0 else 0,
            'prob_0_to_1': flip_0_to_1 / n_flip if n_flip > 0 else 0
        }
    data.loc[flip_idx, 'label'] = 1 - data.loc[flip_idx, 'label']
    print('Flip stats:', flip_stats)
    if seed is not None:
        np.random.seed(MASTER_SEED)
    return data, flip_stats


# ===== VECTORIZERS =====
def create_ngram_vectorizer(ngram_range=(1, 2), max_features=3000):
    return TfidfVectorizer(
        max_features=max_features, tokenizer=ArabicTokenizer(),
        ngram_range=ngram_range, analyzer='word')


# ===== MODELS =====
def get_models(use_class_weights=True, use_real_bert=True):
    cw = 'balanced' if use_class_weights else None
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=500, random_state=MASTER_SEED, class_weight=cw,
            solver='saga', n_jobs=-1),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, random_state=MASTER_SEED, class_weight=cw,
            n_jobs=-1, max_depth=15),
        'SVM': SVC(
            probability=True, random_state=MASTER_SEED, class_weight=cw,
            kernel='linear'),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=300, random_state=MASTER_SEED),
        'Naive Bayes': MultinomialNB(),
        'Arabic BERT' : ArabicBERTModel(
            max_length=64, batch_size=32, epochs=5, learning_rate=3e-5)
    }
    # if BERT_AVAILABLE and use_real_bert:
    #     models['Arabic BERT'] = ArabicBERTModel(
    #         max_length=64, batch_size=32, epochs=5, learning_rate=3e-5)
    #     print("✓ Using optimized Arabic BERT (10 epoch, batch=32)")
    # else:
    #     models['BERT-like'] = SimplifiedBERTModel()
    return models


# ===== METRICS =====
def compute_comprehensive_metrics(y_true, y_pred, y_proba):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['class_f1_not_offensive'] = class_f1[0] if len(class_f1) > 0 else 0.0
    metrics['class_f1_offensive'] = class_f1[1] if len(class_f1) > 1 else 0.0
    pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else np.zeros_like(y_true, dtype=float)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
        metrics['pr_auc'] = average_precision_score(y_true, pos_proba)
    except:
        metrics['roc_auc'] = 0.5
        metrics['pr_auc'] = 0.5
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return metrics


# ============================================================
# ===== FULL ERROR ANALYSIS FUNCTIONS =======================
# ============================================================

def collect_errors(y_true, y_pred, y_proba, X_texts, model_name,
                   noise_level, balance_type, run_id=0):
    """
    Collect detailed information about every misclassified sample.

    Returns a list of dicts, one per error, containing:
    - text, true label, predicted label, confidence scores
    - error type (FP or FN), text length, noise level, etc.
    """
    errors = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    texts = np.array(X_texts)

    pos_conf = y_proba[:, 1] if y_proba.ndim == 2 else np.zeros(len(y_true))

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != yp:
            error_type = 'FP' if (yp == 1 and yt == 0) else 'FN'
            text = str(texts[i])
            token_len = len(text.split())
            char_len = len(text)
            errors.append({
                'run_id': run_id,
                'model': model_name,
                'noise_level': noise_level,
                'balance_type': balance_type,
                'text': text,
                'true_label': int(yt),
                'pred_label': int(yp),
                'error_type': error_type,
                'confidence_offensive': float(pos_conf[i]),
                'confidence_not_offensive': float(1 - pos_conf[i]),
                'token_length': token_len,
                'char_length': char_len,
                'high_confidence_error': float(pos_conf[i]) > 0.8 or float(pos_conf[i]) < 0.2,
            })
    return errors


def analyze_error_patterns(errors_df):
    """
    Analyse collected errors for:
    - Most common misclassified words (top tokens in FP and FN)
    - Length distribution of errors vs correct
    - Confidence distribution of errors
    - High-confidence errors (model very wrong)
    """
    if errors_df.empty:
        return {}

    analysis = {}

    # --- Token frequency in FP vs FN ---
    fp_texts = errors_df[errors_df['error_type'] == 'FP']['text'].tolist()
    fn_texts = errors_df[errors_df['error_type'] == 'FN']['text'].tolist()

    def top_tokens(texts, n=20):
        all_tokens = []
        for t in texts:
            all_tokens.extend(re.findall(r'[\u0600-\u06FF]+', t))
        return Counter(all_tokens).most_common(n)

    analysis['top_tokens_FP'] = top_tokens(fp_texts)
    analysis['top_tokens_FN'] = top_tokens(fn_texts)

    # --- Per-model breakdown ---
    model_breakdown = errors_df.groupby(['model', 'error_type']).size().unstack(fill_value=0)
    analysis['model_breakdown'] = model_breakdown

    # --- Per-noise breakdown ---
    noise_breakdown = errors_df.groupby(['noise_level', 'error_type']).size().unstack(fill_value=0)
    analysis['noise_breakdown'] = noise_breakdown

    # --- High-confidence errors ---
    hc_errors = errors_df[errors_df['high_confidence_error']]
    analysis['high_confidence_count'] = len(hc_errors)
    analysis['high_confidence_pct'] = len(hc_errors) / len(errors_df) * 100

    # --- Average confidence of wrong predictions ---
    analysis['mean_confidence_FP'] = errors_df[errors_df['error_type'] == 'FP'][
        'confidence_offensive'].mean()
    analysis['mean_confidence_FN'] = errors_df[errors_df['error_type'] == 'FN'][
        'confidence_offensive'].mean()

    return analysis


def save_error_examples(errors_df, max_per_condition=50):
    """
    Save CSV tables of error examples for each model × noise × balance condition.
    Each file contains real misclassified texts for qualitative inspection.
    """
    if errors_df.empty:
        print("No errors to save.")
        return

    out_dir = os.path.join(error_dir, 'examples')

    # Full error log (all runs aggregated)
    full_path = os.path.join(error_dir, 'all_errors_log.csv')
    errors_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Full error log → {full_path} ({len(errors_df)} rows)")

    # Per-model × noise condition (sample for readability)
    for (model, noise, balance), grp in errors_df.groupby(
            ['model', 'noise_level', 'balance_type']):
        sample = grp.sample(min(max_per_condition, len(grp)), random_state=MASTER_SEED)
        fname = (f"errors_{model}_{balance}_noise{noise}.csv"
                 .replace(' ', '_').replace('/', '_'))
        sample.to_csv(os.path.join(out_dir, fname), index=False, encoding='utf-8-sig')


def save_error_summary(errors_df):
    """
    Create a high-level summary CSV:
    error rate, FP/FN counts, avg confidence, high-confidence error %
    per model × noise × balance.
    """
    if errors_df.empty:
        return

    rows = []
    for (model, noise, balance), grp in errors_df.groupby(
            ['model', 'noise_level', 'balance_type']):
        n_total = len(grp)
        fp = (grp['error_type'] == 'FP').sum()
        fn = (grp['error_type'] == 'FN').sum()
        hc = grp['high_confidence_error'].sum()
        rows.append({
            'model': model,
            'noise_level': noise,
            'balance_type': balance,
            'total_errors': n_total,
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fp_rate': fp / n_total if n_total else 0,
            'fn_rate': fn / n_total if n_total else 0,
            'high_confidence_errors': int(hc),
            'high_confidence_pct': hc / n_total * 100 if n_total else 0,
            'mean_confidence_FP': grp[grp['error_type'] == 'FP']['confidence_offensive'].mean(),
            'mean_confidence_FN': grp[grp['error_type'] == 'FN']['confidence_offensive'].mean(),
            'mean_token_length_errors': grp['token_length'].mean(),
        })

    summary_df = pd.DataFrame(rows)
    path = os.path.join(error_dir, 'error_summary.csv')
    summary_df.to_csv(path, index=False)
    print(f"  ✓ Error summary → {path}")
    return summary_df


def plot_error_analysis(errors_df, all_raw):
    """
    Generate all error analysis plots:
    1.  FP / FN counts per model (bar, per noise)
    2.  Error count vs noise level (line per model)
    3.  Confidence distribution of errors (KDE per model)
    4.  High-confidence error % vs noise level
    5.  Token-length distribution: errors vs (implied) correct
    6.  FP vs FN heatmap (model × noise)
    7.  Top-20 tokens in FP / FN
    8.  Per-balance error breakdown
    9.  Error rate (fraction wrong) vs noise level
    10. Confusion matrix heatmaps (aggregated)
    """
    if errors_df.empty:
        print("No errors to plot.")
        return

    plot_dir = os.path.join(error_dir, 'plots')

    # ---------- 1. FP / FN bar chart per model ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        sub = errors_df[errors_df['balance_type'] == balance]
        ct = sub.groupby(['model', 'error_type']).size().unstack(fill_value=0)
        ct.plot(kind='bar', ax=ax, color=['#E74C3C', '#3498DB'], edgecolor='black')
        ax.set_title(f'FP vs FN Counts — {balance.title()}')
        ax.set_xlabel('Model')
        ax.set_ylabel('Count (across all runs & noise levels)')
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title='Error Type')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '01_fp_fn_per_model.png'), dpi=200)
    plt.close()

    # ---------- 2. Error count vs noise level ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        sub = errors_df[errors_df['balance_type'] == balance]
        ct = sub.groupby(['noise_level', 'model']).size().reset_index(name='errors')
        sns.lineplot(data=ct, x='noise_level', y='errors', hue='model',
                     marker='o', ax=ax)
        ax.set_title(f'Error Count vs Noise Level — {balance.title()}')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Total Errors')
        ax.set_xticks(NOISE_LEVELS)
        ax.set_xticklabels([f"{int(n*100)}%" for n in NOISE_LEVELS])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02_error_count_vs_noise.png'), dpi=200)
    plt.close()

    # ---------- 3. Confidence distribution (KDE) ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        sub = errors_df[errors_df['balance_type'] == balance]
        for model in sub['model'].unique():
            m_sub = sub[sub['model'] == model]
            if len(m_sub) > 5:
                m_sub['confidence_offensive'].plot.kde(ax=ax, label=model)
        ax.set_title(f'Confidence Score Distribution of Errors — {balance.title()}')
        ax.set_xlabel('P(Offensive) at prediction time')
        ax.legend(title='Model')
        ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '03_confidence_distribution.png'), dpi=200)
    plt.close()

    # ---------- 4. High-confidence error % vs noise ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        sub = errors_df[errors_df['balance_type'] == balance]
        hc_rate = (sub.groupby(['noise_level', 'model'])['high_confidence_error']
                   .mean().reset_index())
        hc_rate['high_confidence_error'] *= 100
        sns.lineplot(data=hc_rate, x='noise_level', y='high_confidence_error',
                     hue='model', marker='o', ax=ax)
        ax.set_title(f'High-Confidence Error % vs Noise — {balance.title()}')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('% of Errors with High Confidence')
        ax.set_xticks(NOISE_LEVELS)
        ax.set_xticklabels([f"{int(n*100)}%" for n in NOISE_LEVELS])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '04_high_confidence_errors_vs_noise.png'), dpi=200)
    plt.close()

    # ---------- 5. Token-length distribution of errors ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        sub = errors_df[errors_df['balance_type'] == balance]
        fp_lens = sub[sub['error_type'] == 'FP']['token_length']
        fn_lens = sub[sub['error_type'] == 'FN']['token_length']
        ax.hist(fp_lens, bins=30, alpha=0.6, label='FP', color='#E74C3C')
        ax.hist(fn_lens, bins=30, alpha=0.6, label='FN', color='#3498DB')
        ax.set_title(f'Token Length of Errors — {balance.title()}')
        ax.set_xlabel('Token Count')
        ax.set_ylabel('Frequency')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '05_token_length_errors.png'), dpi=200)
    plt.close()

    # ---------- 6. FP vs FN heatmap (model × noise) ----------
    for balance in ['unbalanced', 'balanced']:
        sub = errors_df[errors_df['balance_type'] == balance]
        for err_type in ['FP', 'FN']:
            pivot = (sub[sub['error_type'] == err_type]
                     .groupby(['model', 'noise_level']).size()
                     .unstack(fill_value=0))
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title(f'{err_type} Heatmap — {balance.title()}')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Model')
            plt.tight_layout()
            plt.savefig(
                os.path.join(plot_dir, f'06_{err_type}_heatmap_{balance}.png'),
                dpi=200)
            plt.close()

    # ---------- 7. Top tokens in FP and FN ----------
    for balance in ['unbalanced', 'balanced']:
        sub = errors_df[errors_df['balance_type'] == balance]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, err_type, color in zip(axes, ['FP', 'FN'], ['#E74C3C', '#3498DB']):
            texts = sub[sub['error_type'] == err_type]['text'].tolist()
            all_tokens = []
            for t in texts:
                all_tokens.extend(re.findall(r'[\u0600-\u06FF]+', str(t)))
            if all_tokens:
                top = Counter(all_tokens).most_common(20)
                words, counts = zip(*top)
                ax.barh(range(len(words)), counts, color=color)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=9)
                ax.invert_yaxis()
            ax.set_title(f'Top 20 Tokens in {err_type} — {balance.title()}')
            ax.set_xlabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'07_top_tokens_{balance}.png'), dpi=200)
        plt.close()

    # ---------- 8. Per-balance error breakdown (stacked bar by noise) ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        sub = errors_df[errors_df['balance_type'] == balance]
        ct = (sub.groupby(['noise_level', 'error_type'])
              .size().unstack(fill_value=0))
        ct.plot(kind='bar', stacked=True, ax=ax,
                color={'FP': '#E74C3C', 'FN': '#3498DB'}, edgecolor='black')
        ax.set_title(f'Error Composition by Noise — {balance.title()}')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Total Errors')
        ax.set_xticklabels([f"{int(float(str(t).replace('noise',''))*100)}%"
                            if 'noise' not in str(t) else str(t)
                            for t in ct.index], rotation=0)
        ax.legend(title='Error Type')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '08_error_composition_by_noise.png'), dpi=200)
    plt.close()

    # ---------- 9. Error RATE (errors / total) vs noise ----------
    # We need total predictions to compute rate – use all_raw confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        rate_rows = []
        for noise in NOISE_LEVELS:
            model_dict = all_raw.get(balance, {}).get(noise, {})
            for model, met_dict in model_dict.items():
                cms = met_dict.get('confusion_matrix', [])
                if cms:
                    avg_cm = np.mean(cms, axis=0)
                    total = avg_cm.sum()
                    errors_count = total - np.trace(avg_cm)
                    rate_rows.append({
                        'model': model,
                        'noise_level': noise,
                        'error_rate': errors_count / total if total > 0 else 0
                    })
        if rate_rows:
            rate_df = pd.DataFrame(rate_rows)
            sns.lineplot(data=rate_df, x='noise_level', y='error_rate',
                         hue='model', marker='o', ax=ax)
        ax.set_title(f'Error Rate vs Noise — {balance.title()}')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Error Rate (1 - Accuracy)')
        ax.set_xticks(NOISE_LEVELS)
        ax.set_xticklabels([f"{int(n*100)}%" for n in NOISE_LEVELS])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '09_error_rate_vs_noise.png'), dpi=200)
    plt.close()

    # ---------- 10. Per-model FP vs FN ratio ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, balance in zip(axes, ['unbalanced', 'balanced']):
        sub = errors_df[errors_df['balance_type'] == balance]
        ct = sub.groupby(['model', 'error_type']).size().unstack(fill_value=0)
        if 'FP' in ct.columns and 'FN' in ct.columns:
            ct['FP_FN_ratio'] = ct['FP'] / (ct['FN'] + 1e-9)
            ct['FP_FN_ratio'].plot(kind='bar', ax=ax, color='#9B59B6',
                                   edgecolor='black')
        ax.axhline(1.0, color='red', linestyle='--', label='ratio=1 (equal FP/FN)')
        ax.set_title(f'FP/FN Ratio per Model — {balance.title()}')
        ax.set_xlabel('Model')
        ax.set_ylabel('FP / FN Ratio')
        ax.tick_params(axis='x', rotation=30)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '10_fp_fn_ratio.png'), dpi=200)
    plt.close()

    print(f"  ✓ 10 error-analysis plots saved to {plot_dir}/")


def save_per_model_error_reports(errors_df):
    """
    For each model, generate a dedicated report CSV with:
    - per-noise FP/FN breakdown
    - mean confidence per error type per noise
    - top misclassified token patterns
    """
    out_dir = os.path.join(error_dir, 'per_model')
    if errors_df.empty:
        return

    for model in errors_df['model'].unique():
        m_df = errors_df[errors_df['model'] == model]

        rows = []
        for (noise, balance, err_type), grp in m_df.groupby(
                ['noise_level', 'balance_type', 'error_type']):
            all_tokens = []
            for t in grp['text'].tolist():
                all_tokens.extend(re.findall(r'[\u0600-\u06FF]+', str(t)))
            top_tokens_str = ', '.join([w for w, _ in Counter(all_tokens).most_common(10)])

            rows.append({
                'model': model,
                'noise_level': noise,
                'balance_type': balance,
                'error_type': err_type,
                'count': len(grp),
                'mean_confidence': grp['confidence_offensive'].mean(),
                'std_confidence': grp['confidence_offensive'].std(),
                'mean_token_length': grp['token_length'].mean(),
                'high_confidence_errors': grp['high_confidence_error'].sum(),
                'top_10_tokens': top_tokens_str,
            })

        if rows:
            fname = f"report_{model.replace(' ', '_')}.csv"
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, fname), index=False)

    print(f"  ✓ Per-model error reports → {out_dir}/")


def run_full_error_analysis(errors_list, all_raw):
    """
    Master function to orchestrate all error analysis steps.
    Call once after all experiments complete.
    """
    print("\n" + "=" * 60)
    print("📊 RUNNING FULL ERROR ANALYSIS")
    print("=" * 60)

    if not errors_list:
        print("⚠ No errors collected — skipping error analysis.")
        return

    errors_df = pd.DataFrame(errors_list)
    print(f"  Total error records: {len(errors_df)}")

    # 1. Save raw error examples (CSVs)
    print("\n[1/5] Saving error examples...")
    save_error_examples(errors_df)

    # 2. High-level summary
    print("[2/5] Saving error summary...")
    summary = save_error_summary(errors_df)
    if summary is not None:
        print(summary.to_string(index=False))

    # 3. Per-model detailed reports
    print("[3/5] Saving per-model reports...")
    save_per_model_error_reports(errors_df)

    # 4. Pattern analysis (print to console + save)
    print("[4/5] Analysing error patterns...")
    analysis = analyze_error_patterns(errors_df)
    pattern_rows = []
    for key, val in analysis.items():
        if key.startswith('top_tokens'):
            for token, count in val:
                pattern_rows.append({'analysis': key, 'token': token, 'count': count})
    if pattern_rows:
        pd.DataFrame(pattern_rows).to_csv(
            os.path.join(error_dir, 'top_error_tokens.csv'), index=False)
        print(f"  ✓ Token patterns → {error_dir}/top_error_tokens.csv")

    # Print key stats
    print(f"\n  High-confidence errors: {analysis.get('high_confidence_count', 0)} "
          f"({analysis.get('high_confidence_pct', 0):.1f}%)")
    print(f"  Mean conf FP (model predicted Offensive wrongly): "
          f"{analysis.get('mean_confidence_FP', float('nan')):.3f}")
    print(f"  Mean conf FN (model missed Offensive): "
          f"{analysis.get('mean_confidence_FN', float('nan')):.3f}")

    # 5. Plots
    print("[5/5] Generating error analysis plots...")
    plot_error_analysis(errors_df, all_raw)

    print("\n✅ Error analysis complete.")
    print(f"   All outputs → {error_dir}/")


# ============================================================
# ===== EVALUATION (modified to collect errors) ==============
# ============================================================

def evaluate_dataset(X_train_raw, y_train, X_test_raw, y_test,
                     noise_level, seed, is_balanced, run_id=None,
                     save_errors=False, use_real_bert=True,
                     balance_type='unbalanced'):
    """Evaluation with FIXED data preprocessing flow + error collection."""

    train_df = pd.DataFrame({'Comment': X_train_raw.values, 'label': y_train.values})

    if is_balanced:
        indices = np.arange(len(train_df)).reshape(-1, 1)
        ros = RandomOverSampler(random_state=seed)
        indices_resampled, _ = ros.fit_resample(indices, train_df['label'])
        train_df = train_df.iloc[indices_resampled.flatten()].reset_index(drop=True)

    noisy_train, flip_stats = introduce_label_noise(
        train_df, noise_level, seed=seed, track_flips=True)
    X_train_noisy = noisy_train['Comment'].values
    y_train_noisy = noisy_train['label'].values

    vectorizer = create_ngram_vectorizer(ngram_range=(1, 2), max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train_noisy)
    X_test_vec = vectorizer.transform(X_test_raw)

    models = get_models(use_class_weights=True, use_real_bert=use_real_bert)
    results = {}
    collected_errors = []

    for name, model in models.items():
        try:
            if name in ['Arabic BERT', 'BERT-like']:
                model.fit(X_train_noisy, y_train_noisy)
                y_pred = model.predict(X_test_raw)
                y_proba = model.predict_proba(X_test_raw)
                test_texts = X_test_raw
            else:
                model.fit(X_train_vec, y_train_noisy)
                y_pred = model.predict(X_test_vec)
                y_proba = (model.predict_proba(X_test_vec)
                           if hasattr(model, 'predict_proba')
                           else np.zeros((len(y_test), 2)))
                test_texts = X_test_raw  # keep original text for error analysis

            metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba)
            results[name] = metrics
            if flip_stats:
                results[name]['flip_stats'] = flip_stats

            # --- Collect errors ---
            if save_errors:
                errs = collect_errors(
                    y_test, y_pred, y_proba, test_texts,
                    model_name=name,
                    noise_level=noise_level,
                    balance_type=balance_type,
                    run_id=run_id
                )
                collected_errors.extend(errs)

        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {
                'macro_f1': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0,
                'class_f1_offensive': 0.0, 'class_f1_not_offensive': 0.0,
                'roc_auc': 0.0, 'pr_auc': 0.0
            }

    return results, collected_errors


# ===== CV EXPERIMENT =====
def run_experiment_cv(data, use_real_bert=True):
    """OPTIMIZED CV experiment with full error collection."""
    X = data['Comment']
    y = data['label']

    all_raw = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))))
    flip_tracking = defaultdict(lambda: defaultdict(list))

    # Master list for ALL errors across ALL runs
    all_errors = []

    # Strategy: collect errors on ALL folds of repeat 0 only (representative sample)
    # to avoid an enormous CSV. Change COLLECT_ERROR_REPEATS to N_REPEATS for all.
    COLLECT_ERROR_REPEATS = 1  # set to N_REPEATS for exhaustive collection

    total_runs = 2 * len(NOISE_LEVELS) * N_REPEATS * N_SPLITS
    run_counter = 0
    start_time = time.time()

    for is_balanced in [False, True]:
        balance_type = "balanced" if is_balanced else "unbalanced"
        print(f"\n{'=' * 60}")
        print(f"{balance_type.upper()} — {N_REPEATS}x{N_SPLITS} CV")
        print(f"{'=' * 60}")

        for noise in NOISE_LEVELS:
            print(f"\n--- Noise: {noise} ---")

            for repeat in range(N_REPEATS):
                seed = MASTER_SEED + repeat
                skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=seed)

                # Collect errors only for first COLLECT_ERROR_REPEATS repeats
                do_collect = (repeat < COLLECT_ERROR_REPEATS)

                for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                    run_counter += 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / run_counter) * (total_runs - run_counter)
                    print(f"  R{repeat+1}F{fold+1}/{N_SPLITS} "
                          f"[{run_counter}/{total_runs}] "
                          f"ETA: {eta/60:.1f}m", end='\r')

                    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

                    results, fold_errors = evaluate_dataset(
                        X_tr, y_tr, X_te, y_te,
                        noise_level=noise,
                        seed=seed,
                        is_balanced=is_balanced,
                        run_id=repeat * N_SPLITS + fold,
                        save_errors=do_collect,
                        use_real_bert=use_real_bert,
                        balance_type=balance_type
                    )

                    all_errors.extend(fold_errors)

                    for model, metrics in results.items():
                        if 'flip_stats' in metrics and metrics['flip_stats']:
                            flip_tracking[balance_type][noise].append(
                                metrics['flip_stats'])
                        for met, val in metrics.items():
                            if met == 'confusion_matrix':
                                all_raw[balance_type][noise][model][met].append(val)
                            elif met == 'flip_stats':
                                pass
                            elif isinstance(val, (int, float)):
                                all_raw[balance_type][noise][model][met].append(val)

            print()

    total_time = time.time() - start_time
    print(f"\n✓ Total time: {total_time/60:.1f} minutes")

    save_flip_statistics(flip_tracking)

    return all_raw, all_errors


# ===== SAVE FLIP STATS =====
def save_flip_statistics(flip_tracking):
    rows = []
    for balance, noise_dict in flip_tracking.items():
        for noise, stats_list in noise_dict.items():
            if stats_list:
                avg_stats = {k: np.mean([s[k] for s in stats_list])
                             for k in stats_list[0]}
                rows.append({'Balance_Type': balance, 'Noise_Level': noise, **avg_stats})
    if rows:
        df = pd.DataFrame(rows)
        path = f"{results_dir}/flip_statistics.csv"
        df.to_csv(path, index=False)
        print(f"Flip statistics saved to {path}")


# ===== AGGREGATION =====
def save_confusion_matrices(all_raw, suffix=""):
    cm_dir = os.path.join(results_dir, 'confusion_matrices')
    for balance, noise_dict in all_raw.items():
        for noise, model_dict in noise_dict.items():
            for model, met_dict in model_dict.items():
                if 'confusion_matrix' not in met_dict or not met_dict['confusion_matrix']:
                    continue
                cms = met_dict['confusion_matrix']
                avg_cm = np.mean(cms, axis=0) if isinstance(cms[0], np.ndarray) else np.array([[0,0],[0,0]])
                cm_df = pd.DataFrame(avg_cm,
                                     index=['Actual_Not_Offensive', 'Actual_Offensive'],
                                     columns=['Pred_Not_Offensive', 'Pred_Offensive'])
                fname = (f"cm_{balance}_noise{noise}_{model}{suffix}.csv"
                         .replace(" ", "_").replace("/", "_"))
                cm_df.to_csv(os.path.join(cm_dir, fname), index=True)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm_df, annot=True, fmt='.1f', cmap='Blues')
                plt.title(f'{model} – {balance} (Noise={noise})')
                plt.tight_layout()
                plt.savefig(os.path.join(cm_dir, fname.replace('.csv', '.png')), dpi=200)
                plt.close()


def aggregate_and_save_results(all_raw, suffix=""):
    all_runs = []
    for balance, noise_dict in all_raw.items():
        for noise, model_dict in noise_dict.items():
            for model, met_dict in model_dict.items():
                for run_idx, val in enumerate(met_dict.get('macro_f1', [])):
                    run_data = {'Balance_Type': balance, 'Noise_Level': noise,
                                'Model': model, 'Run': run_idx}
                    for met_name, met_vals in met_dict.items():
                        if isinstance(met_vals, list) and len(met_vals) > run_idx:
                            if isinstance(met_vals[run_idx], (int, float)):
                                run_data[met_name] = met_vals[run_idx]
                    all_runs.append(run_data)

    if all_runs:
        df_runs = pd.DataFrame(all_runs)
        df_runs.to_csv(f"{results_dir}/detailed_runs{suffix}.csv", index=False)
        print(f"Detailed runs → {results_dir}/detailed_runs{suffix}.csv")

    agg = defaultdict(lambda: defaultdict(dict))
    for balance, noise_dict in all_raw.items():
        for noise, model_dict in noise_dict.items():
            for model, met_dict in model_dict.items():
                agg[balance][noise][model] = {
                    met: f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                    for met, vals in met_dict.items()
                    if isinstance(vals, list) and vals and isinstance(vals[0], (int, float))
                }

    rows = []
    for balance, noise_dict in agg.items():
        for noise, model_dict in noise_dict.items():
            for model, met_dict in model_dict.items():
                row = {'Balance_Type': balance, 'Noise_Level': noise, 'Model': model}
                row.update(met_dict)
                rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = f"{results_dir}/aggregated_results{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Aggregated results → {csv_path}")

    perform_statistical_tests(all_raw, suffix)
    plot_results(df, suffix)
    save_confusion_matrices(all_raw, suffix=suffix)

    return df


# ===== STATISTICAL TESTS =====
def perform_statistical_tests(all_raw, suffix=""):
    stat_rows = []
    for balance in ['unbalanced', 'balanced']:
        if 0.0 not in all_raw.get(balance, {}):
            continue
        for model in all_raw[balance][0.0].keys():
            f1_0 = all_raw[balance][0.0][model].get('macro_f1', [])
            f1_5 = all_raw[balance].get(0.5, {}).get(model, {}).get('macro_f1', [])
            if len(f1_0) > 1 and len(f1_0) == len(f1_5):
                try:
                    w, p = wilcoxon(f1_0, f1_5, alternative='greater')
                    stat_rows.append({
                        'Balance_Type': balance, 'Model': model,
                        'Comparison': '0.0 vs 0.5 Noise',
                        'Wilcoxon_W': w, 'P_Value': p, 'Significant_Drop': p < 0.05
                    })
                except:
                    pass

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        path = f"{results_dir}/statistical_analysis{suffix}.csv"
        stat_df.to_csv(path, index=False)
        print(f"Statistical tests → {path}")


# ===== PLOTTING =====
def plot_results(df, suffix=""):
    def plot_metric(df, metric='macro_f1'):
        for balance in ['unbalanced', 'balanced']:
            sub = df[df['Balance_Type'] == balance].copy()
            if metric not in sub.columns:
                continue
            sub[f'{metric}_mean'] = sub[metric].apply(lambda x: float(x.split(' ± ')[0]))
            sub[f'{metric}_std'] = sub[metric].apply(lambda x: float(x.split(' ± ')[1]))
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=sub, x='Noise_Level', y=f'{metric}_mean',
                         hue='Model', marker='o')
            for model in sub['Model'].unique():
                md = sub[sub['Model'] == model]
                plt.errorbar(md['Noise_Level'], md[f'{metric}_mean'],
                             yerr=md[f'{metric}_std'], fmt='none', capsize=4, alpha=0.4)
            plt.title(f'{balance.title()} – {metric.replace("_", " ").title()} vs Noise')
            plt.xlabel('Noise Level')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(NOISE_LEVELS, [f"{int(n*100)}%" for n in NOISE_LEVELS])
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig(f"{results_dir}/noise_{metric}_{balance}{suffix}.png", dpi=300)
            plt.close()

    for metric in ['macro_f1', 'macro_precision', 'macro_recall', 'roc_auc', 'pr_auc']:
        if metric in df.columns:
            plot_metric(df, metric)


# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    print("=" * 80)
    print("ARABIC OFFENSIVE LANGUAGE DETECTION — FULL ERROR ANALYSIS")
    print("=" * 80)
    print(f"\nMode: {EXPERIMENT_MODE}  |  CV: {N_REPEATS}x{N_SPLITS}  |  "
          f"Noise levels: {len(NOISE_LEVELS)}")

    USE_REAL_BERT = True
    if USE_REAL_BERT and BERT_AVAILABLE:
        print(f"⚡ BERT device: {DEVICE}")
    elif USE_REAL_BERT and not BERT_AVAILABLE:
        print("⚠ Transformers not available — using simplified BERT.")
        USE_REAL_BERT = False

    print("\n" + "=" * 80)
    data = load_and_preprocess_data()

    print("\n" + "=" * 80)
    if EXPERIMENT_MODE == 'cv':
        print(f"🔄 Starting {N_REPEATS}x{N_SPLITS} Cross-Validation")
        all_results, all_errors = run_experiment_cv(data, use_real_bert=USE_REAL_BERT)
        aggregate_and_save_results(all_results, suffix="_cv")

        # ---- Full error analysis ----
        run_full_error_analysis(all_errors, all_results)

    elif EXPERIMENT_MODE == 'train_test':
        print("Train/Test mode: implement run_experiment_train_test() as needed.")

    if BERT_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared")

    print("\n" + "=" * 80)
    print("✅ EXPERIMENT COMPLETED")
    print("=" * 80)
    suffix = "_cv" if EXPERIMENT_MODE == 'cv' else "_train_test"
    print(f"\nGenerated outputs:")
    print(f"  {results_dir}/aggregated_results{suffix}.csv")
    print(f"  {results_dir}/detailed_runs{suffix}.csv")
    print(f"  {results_dir}/flip_statistics.csv")
    print(f"  {results_dir}/statistical_analysis{suffix}.csv")
    print(f"  {results_dir}/noise_*.png  (performance plots)")
    print(f"\nError analysis outputs ({error_dir}/):")
    print(f"  all_errors_log.csv           ← every misclassified sample")
    print(f"  error_summary.csv            ← FP/FN/confidence per condition")
    print(f"  top_error_tokens.csv         ← most common tokens in FP & FN")
    print(f"  examples/errors_*.csv        ← sampled errors per condition")
    print(f"  per_model/report_*.csv       ← detailed per-model breakdown")
    print(f"  plots/01_fp_fn_per_model.png")
    print(f"  plots/02_error_count_vs_noise.png")
    print(f"  plots/03_confidence_distribution.png")
    print(f"  plots/04_high_confidence_errors_vs_noise.png")
    print(f"  plots/05_token_length_errors.png")
    print(f"  plots/06_FP_heatmap_*.png / 06_FN_heatmap_*.png")
    print(f"  plots/07_top_tokens_*.png")
    print(f"  plots/08_error_composition_by_noise.png")
    print(f"  plots/09_error_rate_vs_noise.png")
    print(f"  plots/10_fp_fn_ratio.png")