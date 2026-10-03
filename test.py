word='Hi'
print(b for b in word.encode("utf-8"))

raw_count = [tuple(bytes([b])) for b in word.encode("utf-8")]