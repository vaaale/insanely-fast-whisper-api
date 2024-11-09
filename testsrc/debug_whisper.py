import torch
import torchaudio
from tokenizers import AddedToken
from transformers import AutoProcessor, WhisperForConditionalGeneration, GenerationConfig, WhisperFeatureExtractor, WhisperTokenizerFast, WhisperTimeStampLogitsProcessor
from transformers import pipeline

audio_file = "/mnt/Data/Datasets/Audio/Stortinget Speach Corpus/data/audio/2010/stortinget-20100106-090925_3240100_3267900.mp3"
model_id = "NbAiLabBeta/nb-whisper-large-semantic"
device = "cuda:0"
torch_dtype = torch.float16

tokenizer = WhisperTokenizerFast.from_pretrained(
    model_id,
    local_files_only=False,
)

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    model_id, local_files_only=False
)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    local_files_only=False,
).to(device)

# model.generation_config.language = "<|no|>"
# model.generation_config.task = "transcribe"

asr = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    model_kwargs={"use_flash_attention_2": False},
    device="cuda:0"
)

result = asr(
    audio_file,
    return_timestamps=True,
    generate_kwargs={
        'task': 'transcribe',
        'language': 'no'
    }
)

# processor = AutoProcessor.from_pretrained(model_id)
# model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
#
# audio_data, sampling_rate = torchaudio.load(audio_file)
# inputs = processor(audio_data[0], return_tensors="pt")
# input_features = inputs.input_features
#
# generated_ids = model.generate(inputs=input_features, return_timestamps=True)
# transcriptions = processor.batch_decode(generated_ids, decode_with_timestamps=True)

print("")
