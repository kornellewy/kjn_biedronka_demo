import os
from difflib import SequenceMatcher


class KjnPostproces(object):
    def __init__(self):
        super().__init__()
        self.rule_for_price_number = {
            'A': '4',
            'g': '9'
        }
        self.rule_for_name_leters = {

        }
        self.treshold_for_sequance_maching = 0.6
        self.rule_for_product_name = {
            'ZAWSZE NISKIE CENY': '',
            'ZAWSZENISKIECENY': '',
            'KUP 2': '',
            'KUP 3': '',
            'KUP2': '',
            'KUP3': '',
            'TANIEJ': '',
            'SUPERCENA': '',
            'CENA': '',
            'ZAWSZE': '',
            'NISKIE CENY': '',
            'SUPER': '',
            'CENA': '',
        }
        self.price_idx_in_zl = 0
        self.price_idx_in_gr = 1
        self.name_of_product_idx = 2
        
    def postproces_list(self, list_to_postproces):
        price_idx_in_zl = self.price_idx_in_zl 
        price_idx_in_gr = self.price_idx_in_gr
        name_of_product_idx = self.name_of_product_idx
        
        dict_with_name_and_prices = {}
        is_readed_priece_is_valid = False
        if not self._list_to_postproces_is_valid(list_to_postproces):
            return is_readed_priece_is_valid, dict_with_name_and_prices

        # postporoces
        try:
            price_in_zl = list_to_postproces[price_idx_in_zl]
            price_in_gr = list_to_postproces[price_idx_in_gr]
            product_name = list_to_postproces[name_of_product_idx]

            # price_in_zl part
            # split price in zl if its longest then 2
            # print("before prepocesing: ", price_in_zl, price_in_gr, product_name)
            if len(price_in_zl) > 2:
                price_in_zl, price_in_gr = self._split_price_in_zl_if_to_long(price_in_zl)
                name_of_product_idx = price_idx_in_gr
                product_name = list_to_postproces[name_of_product_idx]
            # apply transforms for price in zl
            if not self._is_this_zl_price(price_in_zl):
                price_in_zl = self._apply_price_transforms(price_in_zl)

            # price in price_in_gr
            # check if price_in_gr is empty and if next index is valid price_in_gr
            if not self._is_this_gr_price(price_in_gr):
                if price_in_gr == '' and self._is_this_gr_price(product_name):
                    price_in_gr = product_name
                    name_of_product_idx = name_of_product_idx + 1
                    product_name = list_to_postproces[name_of_product_idx]
            # check if price_in_gr is valid and if next index is valid price_in_gr
            if not self._is_this_gr_price(price_in_gr) and self._is_this_gr_price(product_name):
                price_in_gr = product_name
                name_of_product_idx = name_of_product_idx + 1
                product_name = list_to_postproces[name_of_product_idx]
            # check if price_in_gr is valid and if next index is valid price_in_gr after _apply_price_transforms
            if not self._is_this_gr_price(price_in_gr) and self._is_this_gr_price(self._apply_price_transforms(product_name)):
                price_in_gr = self._apply_price_transforms(product_name)
                name_of_product_idx = name_of_product_idx + 1
                product_name = list_to_postproces[name_of_product_idx]
            # aply prepoces to price_in_gr
            if not self._is_this_gr_price(price_in_gr):
                price_in_gr = self._apply_price_transforms(price_in_gr)

            # product name
            # calculare simularity and if 
            for k, v in self.rule_for_product_name.items():
                similari_score = self._calculate_similarity_two_strings_basic(k, product_name)
                if similari_score > self.treshold_for_sequance_maching:
                    product_name = self.rule_for_product_name[k]
            
            # drop empty product name
            if product_name == '':
                name_of_product_idx = name_of_product_idx + 1
                product_name = list_to_postproces[name_of_product_idx]
                print(list_to_postproces)
            
        except (IndexError):
            return is_readed_priece_is_valid, dict_with_name_and_prices

        if self._is_this_zl_price(price_in_zl) and self._is_this_gr_price(price_in_gr) and self._is_this_product_name(product_name):
            is_readed_priece_is_valid = True

        dict_with_name_and_prices = {
            'price_in_zl': price_in_zl,
            'price_in_gr' : price_in_gr,
            'product_name' : product_name,
        }
        return is_readed_priece_is_valid, dict_with_name_and_prices

    def _list_to_postproces_is_valid(self, list_to_postproces):
        if len(list_to_postproces) < 3:
            return False
        return True

    def _split_price_in_zl_if_to_long(self, price_in_zl):
        price_in_zl_cut = price_in_zl[:-2]
        price_in_gr_cut = price_in_zl[-2:]
        return price_in_zl_cut, price_in_gr_cut

    def _is_this_zl_price(self, price_string):
        if len(price_string) < 1 and len(price_string)>2:
            return False
        if not price_string.isdigit():
            return False
        return True

    def _apply_price_transforms(self, price):
        price = self._split(price)
        for idex, i_digit in enumerate(price):
            if i_digit in self.rule_for_price_number:
                price[idex]=self.rule_for_price_number[i_digit]
        price = ''.join(price)
        return price

    def _split(self, word): 
        return [char for char in word]  

    def _is_this_gr_price(self, price_string):
        if len(price_string) !=  2:
            return False
        if not price_string.isdigit():
            return False
        return True
        
    def _calculate_similarity_two_strings_basic(self, a, b):
        # https://docs.python.org/3/library/difflib.html
        return SequenceMatcher(None, a, b).ratio()

    def _is_this_product_name(self, product_name_string):
        if len(product_name_string) < 2:
            return False
        if not product_name_string.isalpha():
            return False
        return True

    def _remove_empty_string(self, string_to_clean):
        while("" in string_to_clean): 
            string_to_clean.remove("")
        return string_to_clean

if __name__ == "__main__":
    import csv
    kjn = KjnPostproces()
    # read csv
    all_readed_prices = []
    with open('cut_pricetags_ordered.csv', 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            all_readed_prices.append(row)
    # remove empty lists
    all_readed_prices = [x for x in all_readed_prices if x != []]

    # and remove 0 idxe form evry list
    for i, x in enumerate(all_readed_prices):
        x = x[1:]
        all_readed_prices[i] = x

    # remove empty lists
    all_readed_prices = [x for x in all_readed_prices if x != []]

    # now fun begin
    for readed_price in  all_readed_prices:
        is_readed_priece_is_valid, dict_with_name_and_pricese = kjn.postproces_list(readed_price)
        print(is_readed_priece_is_valid)
        print(dict_with_name_and_pricese)
        