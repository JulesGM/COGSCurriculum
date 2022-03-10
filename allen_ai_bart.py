import allennlp_models.generation
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.nn.beam_search import BeamSearch
from allennlp.data import Vocabulary
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import ROUGE, BLEU


class BartReuser(allennlp_models.generation.models.Bart):
    def __init__(
        self,
        model,
        model_name: str,
        vocab: Vocabulary,
        max_decoding_steps: int,
        beam_size: int,
    ):
        assert model_name == "facebook/bart-base"
        assert max_decoding_steps == 500, max_decoding_steps
        assert beam_size == 4, beam_size

        super(allennlp_models.generation.models.Bart, self).__init__(vocab)
        self.bart = model
        self._indexer = PretrainedTransformerIndexer(model_name, namespace="tokens")

        self._start_id = self.bart.config.bos_token_id  # CLS
        self._decoder_start_id = self.bart.config.decoder_start_token_id or self._start_id
        self._end_id = self.bart.config.eos_token_id  # SEP
        self._pad_id = self.bart.config.pad_token_id  # PAD

        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_id, max_steps=max_decoding_steps, beam_size=beam_size or 1
        )

        self._rouge = ROUGE(exclude_indices={self._start_id, self._pad_id, self._end_id})
        self._bleu = BLEU(exclude_indices={self._start_id, self._pad_id, self._end_id})
