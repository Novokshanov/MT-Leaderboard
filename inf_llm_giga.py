import os
import re
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import statistics
from itertools import combinations, permutations

'''
lang_dict = {
    'eng_Latn': 'English',
    'fra_Latn': 'French',
    'cmn_Hans': 'Chinese',
    'deu_Latn': 'German',
    'ita_Latn': 'Italian',
    'spa_Latn': 'Spanish',
    'por_Latn_braz1246': 'Portuguese',
    # 'bel_Cyrl': 'Belarusian',
    'ukr_Cyrl': 'Ukranian',
    # 'kir_Cyrl': 'Kyrgyz',
    # 'uzn_Latn': 'Uzbek',
    # 'tgk_Cyrl': 'Tajik',
    # 'azj_Latn': 'Azerbaijani',
    # 'hye_Armn': 'Armenian',
    # 'kaz_Cyrl': 'Kazakh',
    # 'tuk_Latn': 'Turkmen',
    # 'khk_Cyrl': 'Mongolian',
    'tur_Latn': 'Turkish',
    'arz_Arab': 'Modern Standard Arabic', # arb
    'pes_Arab': 'Persian',
    'hin_Deva': 'Hindi',
    # 'heb_Hebr': 'Hebrew',
    # 'prs_Arab': 'Dari',
    # 'pbt_Arab': 'Pashto',
    'jpn_Jpan': 'Japanese',
    'kor_Kore': 'Korean', # Hang
    'tha_Thai': 'Thai',
    'vie_Latn': 'Vietnamese',
    'ind_Latn': 'Indonesian',
    # 'bak_Cyrl': 'Bashkir',
    # 'chv_Cyrl': 'Chuvash',
    # 'myv_Cyrl': 'Erzya',
    # 'tat_Cyrl': 'Tatar',
    # 'ydd_Hebr': 'Yiddish',
    'rus_Cyrl': 'Russian'
}
'''

lang_dict_ru = {
    'eng_Latn': 'английский',
    'fra_Latn': 'французский',
    'cmn_Hans': 'китайский',
    'deu_Latn': 'немекий',
    'ita_Latn': 'итальянский',
    'spa_Latn': 'испанский',
    'por_Latn_braz1246': 'португальский',
    # 'bel_Cyrl': 'Belarusian',
    'ukr_Cyrl': 'украинский',
    # 'kir_Cyrl': 'Kyrgyz',
    # 'uzn_Latn': 'Uzbek',
    # 'tgk_Cyrl': 'Tajik',
    # 'azj_Latn': 'Azerbaijani',
    # 'hye_Armn': 'Armenian',
    # 'kaz_Cyrl': 'Kazakh',
    # 'tuk_Latn': 'Turkmen',
    # 'khk_Cyrl': 'Mongolian',
    'tur_Latn': 'турецкий',
    'arz_Arab': 'арабский', # arb
    'pes_Arab': 'персидский',
    'hin_Deva': 'хинди',
    # 'heb_Hebr': 'Hebrew',
    # 'prs_Arab': 'Dari',
    # 'pbt_Arab': 'Pashto',
    'jpn_Jpan': 'японский',
    'kor_Kore': 'корейский', # Hang
    'tha_Thai': 'тайский',
    'vie_Latn': 'вьетнамский',
    'ind_Latn': 'индонезийский',
    # 'bak_Cyrl': 'Bashkir',
    # 'chv_Cyrl': 'Chuvash',
    # 'myv_Cyrl': 'Erzya',
    # 'tat_Cyrl': 'Tatar',
    # 'ydd_Hebr': 'Yiddish',
    'rus_Cyrl': 'русский'
}


def get_promts(tokenizer, texts, src_lang, tgt_lang):
    prompts = []
    for txt in texts:
        promt = f"Язык следующего текста -- {lang_dict_ru[src_lang]}. Переведи текст на {lang_dict_ru[tgt_lang]} без дополнительных комментариев, не меняя структуру текста. Текст для перевода:\n{txt}\nПеревод: "
        messages = [
            # {"role": "system", "content": promt},
            {"role": "user", "content": promt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        textP = TextPrompt(prompt=text)
        prompts.append(textP)
    return prompts


if __name__ == "__main__":
    model_name = "ai-sage/GigaChat-20B-A3B-instruct"
    llm = LLM(model=model_name, gpu_memory_utilization=0.9, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.3)

    promts = []
    directions = []
    for lang in list(lang_dict_ru.keys())[:-1]:
        for src_lang, tgt_lang in list(permutations(['rus_Cyrl', lang], 2)):
            direction = src_lang + '_2_' + tgt_lang
            inputs = load_dataset("facebook/bouquet", src_lang, split='test')['src_text']
            inputs = [str(x) for x in inputs]
            dir_promts = get_promts(tokenizer, inputs, src_lang, tgt_lang)
            promts.extend(dir_promts)
            directions.extend([direction] * len(dir_promts))


    outputs = llm.generate(promts, sampling_params)
    translations = [output.outputs[0].text for output in outputs]

    df = pd.DataFrame(data={
        'directions': directions,
        'translations': translations
        }
    )

    df.to_csv("bouquet/bouquet-gigachat_20b_A3B_instruct.csv", index=False, sep='|')

    print("Inference finished.")
