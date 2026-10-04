import pandas as pd

import predict
import train_model


def test_stemming_strips_non_letters_and_stopwords():
    assert train_model.stemming("I am LOVING this!!! :) http://t.co") == "love http co"


def test_stemming_empty_text():
    assert train_model.stemming("") == ""


def test_load_dataset_maps_sentiment140_labels(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text('"0","1","d","NO_QUERY","u","sad day"\n"4","2","d","NO_QUERY","u","happy day"\n',
                   encoding="ISO-8859-1")
    df = train_model.load_dataset(str(csv))
    assert list(df["target"]) == [0, 1]
    assert list(df["text"]) == ["sad day", "happy day"]


def test_train_save_predict_roundtrip(tmp_path):
    pos = ["I love this, great day", "happy happy wonderful", "awesome fun amazing", "best thing ever love"]
    neg = ["I hate this, awful day", "sad terrible horrible", "worst thing ever hate", "angry bad miserable"]
    df = pd.DataFrame({"text": (pos + neg) * 5, "target": ([1] * 4 + [0] * 4) * 5})
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    model, vectorizer, train_acc, _test_acc = train_model.train(df)
    assert train_acc == 1.0

    model_path, vec_path = tmp_path / "m.sav", tmp_path / "v.sav"
    train_model.save(model, vectorizer, str(model_path), str(vec_path))
    model, vectorizer = predict.load(str(model_path), str(vec_path))
    assert predict.predict(["love this wonderful day", "hate this awful day"], model, vectorizer) == [
        "Positive", "Negative"]


def test_predict_cli_reports_missing_model(tmp_path, capsys):
    rc = predict.main(["hello", "--model", str(tmp_path / "missing.sav")])
    assert rc == 1
    assert "Run `python train_model.py` first" in capsys.readouterr().err
