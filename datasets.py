import copy
import random
import torch

from torch.utils.data import Dataset
from utils import generate_rating_matrix_valid, generate_rating_matrix_test


# Intent Data Segmentation operations
def IDS(i_file,o_file,max_len):
    """
    :param i_file: original data
    :param o_file: output data
    :max_len: the max length of the sequence
    :return:
    """
    with open(i_file,"r+") as fr:
        data=fr.readlines()
    aug_d={}
    # training, validation, and testing
    max_save_len=max_len+3
    # save
    max_keep_len=max_len+2
    for d_ in data:
        u_i,item=d_.split(' ',1)
        item=item.split(' ')
        item[-1]=str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start=0
        j=3
        if len(item)>max_save_len:
            # training, validation, and testing
            while start<len(item)-max_keep_len:
                j=start+4
                while j<len(item):
                    if start<1 and j-start<max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j+=1
                    else:
                        aug_d[u_i].append(item[start:start+max_save_len])
                        break
                start+=1
        else:
            while j<len(item):
                aug_d[u_i].append(item[start:j+1])
                j+=1
    with open(o_file,"w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i+" "+' '.join(i_)+"\n")

def IDS_random(i_file, o_file, max_len):
    """
    :param i_file: original data
    :param o_file: output data
    :param max_len: the max length of the sequence
    :return:
    """
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    
    aug_d = {}
    # training, validation, and testing
    max_save_len = max_len + 3

    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))  
        aug_d.setdefault(u_i, [])
        
        start = 0
        j = 3
        if len(item) > max_save_len:
            # training, validation, and testing
            while start < len(item) - (max_len + 2):
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        random_sample = random.sample(item[start:j], len(item[start:j]))
                        aug_d[u_i].append(random_sample)
                        j += 1
                    else:
                        random_sample = random.sample(item[start:start + max_save_len], len(item[start:start + max_save_len]))
                        aug_d[u_i].append(random_sample)
                        break
                start += 1
        else:
            while j < len(item):
                random_sample = random.sample(item[start:j + 1], len(item[start:j + 1]))
                aug_d[u_i].append(random_sample)
                j += 1

    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")


def get_seqs_and_matrixes(type, data_file):
    user_seq = []
    user_id = []
    item_set = set()
    
    with open(data_file, "r") as fr:
        for line in fr:
            parts = line.strip().split()
            user = int(parts[0])
            items = list(map(int, parts[1:]))
            user_id.append(user)
            user_seq.append(items)
            item_set.update(items)
    
    max_item = max(item_set)
    num_users = len(user_id)
    num_items = max_item + 2
    
    if type == "training":
        return user_id, user_seq
    elif type == "rating":
        valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
        test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
        return user_id, user_seq, max_item, valid_rating_matrix, test_rating_matrix
    else:
        raise NotImplementedError


class DatasetForVQICL(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length


    def __getitem__(self, index):
        user_id = index
        item_seq = self.user_seq[user_id]

        if self.data_type == "train":
            input_seq = item_seq[ : -3]
            target_pos = item_seq[1 : -2]
            answer = item_seq[-3]

        elif self.data_type == "valid":
            input_seq = item_seq[:-2]
            target_pos = item_seq[1:-1]
            answer = [item_seq[-2]]
        
        else:
            item_seq_with_noise = self._add_noise_for_robustness(item_seq)
            input_seq = item_seq_with_noise[:-1]
            target_pos = item_seq_with_noise[1:]
            answer = [item_seq_with_noise[-1]]

        cur_rec_tensors = self._data_construction(user_id, input_seq, target_pos, answer)

        return cur_rec_tensors


    # padding and to tensor
    def _data_construction(self, user_id, input_seq, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_seq = copy.deepcopy(input_seq)
        input_seq_len = len(copied_input_seq)
        pad_len = self.max_len - input_seq_len
        copied_input_seq =[0] * pad_len+copied_input_seq
        copied_input_seq=copied_input_seq[-self.max_len:]

        # padding
        target_pos =  [0] * pad_len+target_pos
        target_pos = target_pos[-self.max_len:]

        assert len(target_pos) == self.max_len
        assert len(copied_input_seq) == self.max_len

        # to tensor
        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_seq, dtype=torch.long),
            torch.tensor(input_seq_len, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _add_noise_for_robustness(self, items):
        if self.args.noise_ratio == 0:
            return items
        copied_sequence = copy.deepcopy(items)
        insert_nums = int(self.args.noise_ratio * len(copied_sequence))
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)

