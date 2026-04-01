"""
Chạy ứng dụng Streamlit
"""

import os
import subprocess
import sys

def main():
    print("="*60)
    print("🐦 BIRD SEARCH SYSTEM - STREAMLIT APP")
    print("="*60)
    print()
    
    # Kiểm tra xem database đã được xây dựng chưa
    if not os.path.exists("database/search_index.pkl"):
        print("❌ Database chưa được xây dựng!")
        print("\nChạy lệnh sau để xây dựng database:")
        print("   python build_database_script.py")
        return False
    
    print("✅ Database đã sẵn sàng")
    print("\n🚀 Khởi động ứng dụng Streamlit...")
    print("\nTruy cập ứng dụng tại: http://localhost:8501")
    print("\nNhấn Ctrl+C để dừng ứng dụng")
    print("-" * 60)
    print()
    
    # Chạy streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], 
                       check=False)
    except KeyboardInterrupt:
        print("\n\n✅ Ứng dụng đã dừng")
        return True
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
