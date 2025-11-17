from comet import download_model, load_from_checkpoint
from pandas.errors import ParserError
from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import statistics
import evaluate
import torch
import jieba
import glob
import csv
import os

torch.set_float32_matmul_precision('high')

lang_dict = {
    'eng_Latn': '',
    'fra_Latn': '',
    'cmn_Hans': 'zho_Hans', # zho_Hans in nllb project and original dataset
    'deu_Latn': '',
    'ita_Latn': '',
    'spa_Latn': '',
    'por_Latn': 'por_Latn_braz1246',
    'bel_Cyrl': '',
    'ukr_Cyrl': '',
    'kir_Cyrl': '',
    'uzn_Latn': '',
    'tgk_Cyrl': '',
    'azj_Latn': '',
    'hye_Armn': '',
    'kaz_Cyrl': '',
    'tuk_Latn': '',
    'khk_Cyrl': '',
    'tur_Latn': '',
    'arb_Arab': 'arz_Arab',
    'pes_Arab': '',
    'hin_Deva': '',
    'heb_Hebr': '',
    'prs_Arab': '',
    'pbt_Arab': '',
    'jpn_Jpan': '',
    'kor_Hang': 'kor_Kore',
    'tha_Thai': '',
    'vie_Latn': '',
    'ind_Latn': '',
    'bak_Cyrl': '',
    'chv_Cyrl': '',
    'myv_Cyrl': '',
    'tat_Cyrl': '',
    'ydd_Hebr': '',
    'rus_Cyrl': ''
}

def chinese_bleu_meteor(bleu, meteor, preds: list, refs: list):
    meteors = []
    bleus = []
    for i in range(0, len(preds)):
        pred_tokens = jieba.lcut(preds[i])
        ref_tokens = jieba.lcut(refs[i])
        meteors.append(meteor.compute(predictions=[' '.join(pred_tokens)], references=[' '.join(ref_tokens)])['meteor'])
        bleus.append(bleu.compute(predictions=[' '.join(pred_tokens)], references=[' '.join(ref_tokens)])['bleu'])

    return statistics.mean(bleus), statistics.mean(meteors)


def score_direction(bleu, chrf, meteor, comet, xcomet, inputs, preds, refs, tgt_lang):
    chrf_score = chrf.compute(predictions=preds, references=refs)['score']
    chrf_plus_score = chrf.compute(predictions=preds, references=refs, word_order=2)['score']

    if tgt_lang == 'cmn_Hans' or tgt_lang == 'zho_Hans':
        bleu_score, meteor_score = chinese_bleu_meteor(bleu, meteor, preds, refs)
    else:
        bleu_score = bleu.compute(predictions=preds, references=refs)['bleu']
        meteor_score = meteor.compute(predictions=preds, references=refs)['meteor']

    comet_score = comet.predict(
        [{'src': src, 'ref': ref, 'mt': mt} for src, ref, mt in zip(inputs,
                                                                    refs,
                                                                    preds)],
        batch_size=1,
        gpus=1,
        # devices=[2]
    )[1]

    xcomet_score = xcomet.predict(
        [{'src': src, 'ref': ref, 'mt': mt} for src, ref, mt in zip(inputs,
                                                                    refs,
                                                                    preds)],
        batch_size=1,
        gpus=1,
        # devices=[2]
    )[1]
    results = [round(x, 4) for x in [bleu_score, chrf_score, chrf_plus_score, float(meteor_score), comet_score, xcomet_score]]
    return results

def main():
    meteor = evaluate.load("meteor")
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    path_comet = download_model("Unbabel/wmt22-comet-da")
    path_xcomet = download_model("Unbabel/XCOMET-XXL")
    xcomet = load_from_checkpoint(path_xcomet)
    comet = load_from_checkpoint(path_comet)

    DIRPATH = "./flores_plus"
    csv_files = sorted(glob.glob(os.path.join(DIRPATH, '**', '*.csv'), recursive=True))

    DIRPATH_OUT = DIRPATH + "-scores"
    os.makedirs(DIRPATH_OUT, exist_ok=True)

    for file in tqdm(csv_files[1:]):
        try:
            df = pd.read_csv(file)
        except ParserError:
            df = pd.read_csv(file, sep='|')
        print(file)
        directions = df['directions'].unique().tolist()
        filepath_scored = os.path.join(DIRPATH_OUT, Path(file).stem + '-scores.csv')

        with open(filepath_scored, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Direction', 'BLEU', 'chrF', 'chrF++', 'Meteor', 'Comet-wmt22', 'XComet-XXL'])

        for direction in tqdm(directions):
            src_lang = direction.split('_2_')[0]
            tgt_lang = direction.split('_2_')[1]

            preds = df.loc[df['directions'] == direction]['translations'].tolist()
            preds = [str(x) for x in preds]
            try:
                inputs = load_dataset("facebook/bouquet", src_lang, split='test')['src_text']
            except Exception:
                inputs = load_dataset("facebook/bouquet", lang_dict[src_lang], split='test')['src_text']
                
            try:
                refs = load_dataset("facebook/bouquet", tgt_lang, split='test')['src_text']
            except Exception:
                refs = load_dataset("facebook/bouquet", lang_dict[tgt_lang], split='test')['src_text']

            scores = score_direction(bleu, chrf, meteor, comet, xcomet, inputs, preds, refs, tgt_lang)

            with open(filepath_scored, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([direction] + scores)
        
        print("Scoring finished.")

if __name__ == "__main__":
    main()
