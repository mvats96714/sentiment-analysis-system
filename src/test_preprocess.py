from preprocess import clean_text

sample = "I absolutely loved this movie! <br /> It was AMAZING!!! 10/10"

print("Original:")
print(sample)

print("\nCleaned:")
print(clean_text(sample))