import pandas as pd
import re

def clean_name(name, output_file="cleaned_names.txt", dump_to_file=True):
    """
    Clean a name and optionally dump the result to a file.
    
    Args:
        name: Input name to clean
        output_file: File path to dump cleaned names (default: "cleaned_names.txt")
        dump_to_file: Whether to dump results to file (default: True)
    
    Returns:
        Cleaned name or None if invalid
    """
    if not isinstance(name, str):
        cleaned = None
    else:
        first_word = name.split(' ')[0]
        if re.fullmatch(r'[a-z]+', first_word):
            cleaned = first_word
        else:
            cleaned = None
    
    # Dump to file if requested
    if dump_to_file and cleaned is not None:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"{cleaned}\n")
    
    return cleaned

def clean_names_batch(names_list, output_file="cleaned_names_batch.txt", clear_file=True):
    """
    Clean a batch of names and dump all valid results to a file.
    
    Args:
        names_list: List of names to clean
        output_file: File path to dump cleaned names
        clear_file: Whether to clear the file before writing (default: True)
    
    Returns:
        List of cleaned names (None values filtered out)
    """
    # Clear the file if requested
    if clear_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")  # Clear file
    
    cleaned_names = []
    valid_count = 0
    
    for name in names_list:
        cleaned = clean_name(name, output_file, dump_to_file=True)
        if cleaned is not None:
            cleaned_names.append(cleaned)
            valid_count += 1
    
    # Add summary to file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n# Summary: {valid_count} valid names out of {len(names_list)} total\n")
    
    print(f"✅ Dumped {valid_count} cleaned names to {output_file}")
    return cleaned_names

# Example usage:
if __name__ == "__main__":
    # Clear any existing output files
    open("cleaned_names.txt", 'w').close()
    
    try:
        df = pd.read_csv("India_names.csv")  # replace with your CSV file path
        
        # Method 1: Individual name cleaning with file dump
        print("🧹 Method 1: Individual cleaning with file dump")
        df['clean_name'] = df['name'].apply(lambda x: clean_name(x, "cleaned_names.txt"))
        print(df[['name', 'clean_name']].head(10))
        
        # Method 2: Batch cleaning with file dump
        print("\n🧹 Method 2: Batch cleaning with file dump")
        names_list = df['name'].tolist()
        cleaned_batch = clean_names_batch(names_list, "cleaned_names_batch.txt")
        print(f"Processed {len(names_list)} names, got {len(cleaned_batch)} valid names")
        
        print(f"\n📁 Output files created:")
        print(f"   - cleaned_names.txt (individual dumps)")
        print(f"   - cleaned_names_batch.txt (batch dump)")
        
    except FileNotFoundError:
        print("⚠️  India_names.csv not found. Testing with sample data...")
        
        # Test with sample data
        sample_names = ["john doe", "mary", "INVALID123", "alice wonderland", "bob", "charlie brown", "ALLCAPS"]
        print(f"\n🧪 Testing with sample names: {sample_names}")
        
        # Test batch processing
        cleaned_batch = clean_names_batch(sample_names, "sample_cleaned_names.txt")
        print(f"Sample results: {cleaned_batch}")
        print("📁 Check sample_cleaned_names.txt for dumped output")
