# Selection data and provenance

The approved additions were selected using speech transcripts and written text
for 22 Indic profiles. The frozen study contained 280,471 training, 33,027
development and 20,055 reserved records. It combined IndicVoices conversation,
extempore and read speech with SPRING, Omnilingual ASR, UrduSpeech and written
sources. Source and language weighting limited dominance by larger collections.

Candidate strings came from corpus fitting, IndicConformer and IndicBART or
IndicBARTSS. Donor scores and normalizers were not copied. Scores were fitted
against the unchanged native Nemotron vocabulary and pruned to the approved
piece budgets. The released inventories are a selected result, not a claim of
global optimality or complete dialect coverage.

The reserve records all come from IndicVoices. Nine profiles had only one
independent speech origin. Development and reserve were held out from this
fitting process, but independence from donor or native-model pretraining is not
established. Vaani was inspected but excluded because lexical extraction was
unresolved; IN22-Conv was unavailable and excluded.

Current development and reserve coverage is recorded in
[language coverage](language-coverage.md). Rebuilding the current tokenizer does
not download these datasets or rerun selection. No corpus text or audio is
included in this repository.

The immutable [data preparation record](https://github.com/plivo-labs/untok/blob/e2f8acd1bab8edf1dc72267678d4a2ba41949439/docs/native-unigram-data.md)
contains sampling, annotation cleanup, deduplication, overlap handling and source
limitations. Its [fitting results](https://github.com/plivo-labs/untok/blob/e2f8acd1bab8edf1dc72267678d4a2ba41949439/docs/native-unigram-results.md)
belong to the historical artifacts identified there. Keep the retained source
locks and [component licenses](../THIRD_PARTY.md) when using that provenance.
