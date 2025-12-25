import json

# Load phytochemical data
with open('data/phytochemical_mapping.json', 'r') as f:
    data = json.load(f)

# Count compounds per plant
counts = {plant: len(info['phytochemicals']) for plant, info in data.items()}

print("=" * 60)
print("GEMINI API CALL ANALYSIS")
print("=" * 60)

print(f"\n📊 DATASET STATISTICS:")
print(f"   Total plants in database: {len(counts)}")
print(f"   Total compounds across all plants: {sum(counts.values())}")

print(f"\n📈 COMPOUNDS PER PLANT:")
print(f"   Minimum: {min(counts.values())} compounds")
print(f"   Maximum: {max(counts.values())} compounds")
print(f"   Average: {sum(counts.values())/len(counts):.1f} compounds")

print(f"\n🔍 TOP 10 PLANTS BY COMPOUND COUNT:")
for plant, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"   {plant}: {count} compounds")

print(f"\n" + "=" * 60)
print("GEMINI API CALLS PER USER INTERACTION")
print("=" * 60)

print(f"\n🎯 SCENARIO 1: Single Leaf Prediction (/predict endpoint)")
print(f"   When user uploads a leaf image:")
print(f"   - System identifies the plant")
print(f"   - For EACH phytochemical compound, 1 Gemini API call is made")
print(f"   ")
print(f"   Example: If 'Tulsi' is identified:")
tulsi_count = counts.get('Tulsi', 0)
print(f"   - Tulsi has {tulsi_count} compounds")
print(f"   - Gemini API calls: {tulsi_count} calls")
print(f"   ")
print(f"   Another example: If 'Neem' is identified:")
neem_count = counts.get('Neem', 0)
print(f"   - Neem has {neem_count} compounds")
print(f"   - Gemini API calls: {neem_count} calls")

print(f"\n🎯 SCENARIO 2: Manual Description Request (/get_ai_description endpoint)")
print(f"   When user clicks 'Generate AI Description' button:")
print(f"   - 1 Gemini API call per compound")

print(f"\n" + "=" * 60)
print("TOTAL API USAGE SUMMARY")
print("=" * 60)

print(f"\n💡 PER LEAF UPLOAD:")
print(f"   Minimum possible calls: {min(counts.values())} (for plants with fewest compounds)")
print(f"   Maximum possible calls: {max(counts.values())} (for plants with most compounds)")
print(f"   Average calls per upload: {sum(counts.values())/len(counts):.1f}")

print(f"\n📊 DAILY USAGE ESTIMATES:")
daily_limit = 1500
print(f"   Gemini Free Tier Limit: {daily_limit} requests/day")
avg_compounds = sum(counts.values())/len(counts)
print(f"   Maximum leaf predictions per day: ~{int(daily_limit/avg_compounds)} uploads")
print(f"   (assuming average {avg_compounds:.1f} compounds per plant)")

print(f"\n⚠️  IMPORTANT NOTES:")
print(f"   1. Each compound description = 1 API call")
print(f"   2. Calls are made during /predict endpoint processing")
print(f"   3. No caching is currently implemented")
print(f"   4. Failed calls are tracked but not retried")

print("\n" + "=" * 60)
