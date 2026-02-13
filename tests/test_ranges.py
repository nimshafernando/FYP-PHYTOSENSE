#!/usr/bin/env python3
"""
Test Updated IC50 Ranges

Verify that all compounds now fall within the specified ranges.
"""

from reference_ic50_data import REFERENCE_IC50_DATA, generate_closest_ic50_value

def test_ranges():
    """Test that generated IC50 values fall within expected ranges"""
    
    print("🧪 TESTING UPDATED IC50 RANGES")
    print("=" * 60)
    print(f"{'Compound':<15} | {'Expected Range':<20} | {'Generated Values':<25} | {'Status'}")
    print("-" * 85)
    
    # Expected ranges (convert nM to µM where needed)
    expected_ranges = {
        "vincristine": (0.001, 0.010),   # 1–10 nM
        "vinblastine": (0.001, 0.006),   # 1–6 nM
        "capsaicin": (70, 100),          # 70–100 µM
        "curcumin": (40, 90),            # 40–90 µM
        "berberine": (50, 250),          # 50–250 µM
        "piperine": (40, 50),            # 40–50 µM
        "luteolin": (90, 120),           # 90–120 µM
        "quercetin": (120, 150),         # 120–150 µM
        "baicalein": (150, 180),         # 150–180 µM
        "apigenin": (200, 260),          # 200–260 µM
        "egcg": (800, 900),              # 800–900 µM
        "daidzein": (850, 1700)          # 850–1700 µM
    }
    
    all_passed = True
    
    for compound, (min_range, max_range) in expected_ranges.items():
        # Generate 5 test values
        test_values = []
        for _ in range(5):
            value = generate_closest_ic50_value(compound)
            test_values.append(value)
        
        # Check if all values fall within range
        in_range = all(min_range <= val <= max_range for val in test_values)
        status = "✅ PASS" if in_range else "❌ FAIL"
        
        if not in_range:
            all_passed = False
        
        # Format range display
        if min_range < 1:
            range_str = f"{min_range*1000:.0f}-{max_range*1000:.0f} nM"
        else:
            range_str = f"{min_range:.0f}-{max_range:.0f} µM"
        
        # Format test values
        values_str = ", ".join([f"{val:.4f}" if val < 1 else f"{val:.0f}" for val in test_values])
        
        print(f"{compound:<15} | {range_str:<20} | {values_str:<25} | {status}")
    
    print("-" * 85)
    print(f"\n{'✅ ALL RANGES CORRECT' if all_passed else '❌ SOME RANGES NEED ADJUSTMENT'}")
    
    if all_passed:
        print("\n🎯 PERFECT! All compounds now generate IC50 values within your specified ranges.")
        print("   • Vincristine & Vinblastine: Nanomolar range (very potent)")
        print("   • Capsaicin to Apigenin: Micromolar range (moderate activity)")  
        print("   • EGCG & Daidzein: High micromolar range (weak activity)")
    else:
        print("\n⚠️  Some compounds may need range adjustment.")

if __name__ == "__main__":
    test_ranges()