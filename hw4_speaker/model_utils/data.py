import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence










class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # Load the mapping from speaker neme to their corresponding id.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num


path = '/home/dataset/lhy/hw4/Dataset'

def readTrainMel(dataDir):
    map_path = Path(dataDir) / 'mapping.json'
    mapping = json.load(map_path.open())
    print(mapping)
    speaker2id, id2speaker = mapping.values()
    x = []
    y = []
    val_x = []
    val_y = []
    dataPath = Path(dataDir) / 'metadata.json'
    data = json.load(dataPath.open())
    speakers = data['speakers']
    for speaker in speakers:
        speakerList = speakers[speaker]
        # for each in speakerList:
        #     tmp = random.randint(0,10)
        #     if tmp == 1:
        #         val_x.append((each['feature_path'], each['mel_len']))
        #         val_y.append(speaker2id[speaker])
        #     else:
        #         x.append((each['feature_path'], each['mel_len']))
        #         y.append(speaker2id[speaker])
        # return x, y, val_x, val_y
        for each in speakerList:
            x.append((each['feature_path'], each['mel_len']))
            y.append(speaker2id[speaker])
    return x, y


class speakerData(Dataset):
    def __init__(self, dataDir, mode='train', segment_len=256, x=None , y= None):
        super(speakerData, self).__init__()
        self.segment_len = segment_len
        self.dataDir = dataDir
        self.mode = mode
        map_path = Path(dataDir) / 'mapping.json'
        mapping = json.load(map_path.open())
        print(mapping)
        speaker2id, id2speaker = mapping.values()
        if mode == 'train':
            self.x = x
            self.y = torch.LongTensor(y)
        elif mode == 'test':
            x = []
            dataPath = Path(dataDir) / 'testdata.json'
            data = json.load(dataPath.open())
            n_mels = data['n_mels']
            utterances = data['utterances']
            for utterance in utterances:
                x.append((utterance['feature_path'], utterance['mel_len']))
            self.x = x
            self.id2speaker = id2speaker

    def __getitem__(self, item):
        uttrPath = os.path.join(self.dataDir ,self.x[item][0])
        uttrData = torch.load(uttrPath)
        n_mel = self.x[item][1]
        if self.mode == 'train':
            if n_mel > self.segment_len:
                start = random.randint(0, n_mel-self.segment_len)
                end = start + self.segment_len
                uttrData = uttrData[start: end]
        if self.mode == 'test':
            segment_len = 2000
            if n_mel > segment_len:
                start = random.randint(0, n_mel-self.segment_len)
                end = start + self.segment_len
                uttrData = uttrData[start: end]
        if self.mode == 'train':
            return uttrData, self.y[item]
        elif self.mode == 'test':
            return self.x[item][0], uttrData

    def __len__(self):
        return len(self.x)

def my_clooate(batch):
    mel, speaker = zip(*batch)
    mel = pad_sequence(mel,batch_first=True,padding_value=-20)
    return mel, torch.LongTensor(speaker)


def getDataloader(path, batchSize, mode):
    if mode == 'train' :
        # x, y, val_x, val_y = readTrainMel(path)
        # train_set = speakerData(path, mode=mode, x=x, y=y)
        # val_set = speakerData(path, mode=mode, x=val_x, y=val_y)
        # train_loader = DataLoader(train_set, batchSize, shuffle=True,collate_fn=my_clooate)
        # val_loader = DataLoader(val_set, batchSize, shuffle=True,collate_fn=my_clooate)
        # return train_loader, val_loader
        x , y = readTrainMel(path)
        dataset = speakerData(path, mode='train', x=x, y=y)
        trainlen = int(0.9 * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        train_set, val_set = random_split(dataset, lengths)
        train_loader = DataLoader(train_set, batchSize, shuffle=True,collate_fn=my_clooate)
        val_loader = DataLoader(val_set, batchSize, shuffle=True,collate_fn=my_clooate)
        return train_loader, val_loader
    elif mode == 'test':
        dataset = speakerData(path, mode=mode)
        test_loader = DataLoader(dataset, 1, shuffle=False,collate_fn=inference_collate_batch)
        return test_loader

def collate_batch_test(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()

def get_dataloader_test(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch_test,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch_test,
    )

    return train_loader, valid_loader, speaker_num



class myDataset_test(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # Load the mapping from speaker neme to their corresponding id.
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)

def getInferenceDataloader(datapath):
    dataset = InferenceDataset(datapath)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        collate_fn=inference_collate_batch,
    )
    return dataloader



if __name__ == '__main__':
    print(1)

