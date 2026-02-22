import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import glob

# For file reading
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyPDF2 not available. PDF reading disabled.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not available. DOCX reading disabled.")

# For OCR
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: PIL/pytesseract not available. OCR features disabled.")


class AssignmentChecker:
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize the Assignment Checker with Pandas DataFrames
        
        Args:
            similarity_threshold: Threshold for flagging similar submissions (0-1)
        """
        self.similarity_threshold = similarity_threshold
        
        # Use Pandas DataFrames for better data management
        self.submissions_df = pd.DataFrame(columns=[
            'student_id', 'content', 'filename', 'filepath', 'filetype',
            'timestamp', 'hash', 'word_count', 'char_count'
        ])
        
        self.grades_df = pd.DataFrame(columns=[
            'student_id', 'total_score', 'max_score', 'percentage', 
            'letter_grade', 'timestamp'
        ])
        
        self.plagiarism_df = pd.DataFrame(columns=[
            'student_1', 'student_2', 'similarity_score', 'timestamp'
        ])
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
    
    def read_txt_file(self, filepath: str) -> str:
        """Read text from TXT file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading TXT: {str(e)}"
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    
    def read_pdf_file(self, filepath: str) -> str:
        """Read text from PDF file"""
        if not PDF_AVAILABLE:
            return "Error: PyPDF2 not installed"
        
        try:
            text = []
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def read_docx_file(self, filepath: str) -> str:
        """Read text from DOCX file"""
        if not DOCX_AVAILABLE:
            return "Error: python-docx not installed"
        
        try:
            doc = Document(filepath)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def read_image_file(self, filepath: str) -> str:
        """Read text from image using OCR"""
        if not OCR_AVAILABLE:
            return "Error: OCR libraries not installed"
        
        try:
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            return f"Error performing OCR: {str(e)}"
    
    def read_file(self, filepath: str) -> Tuple[str, str]:
        """
        Automatically detect file type and read content
        
        Args:
            filepath: Path to the file
        
        Returns:
            Tuple of (content, filetype)
        """
        if not os.path.exists(filepath):
            return f"Error: File not found - {filepath}", "unknown"
        
        ext = os.path.splitext(filepath)[1].lower()
        
        # Text files
        if ext == '.txt':
            return self.read_txt_file(filepath), 'txt'
        
        # PDF files
        elif ext == '.pdf':
            return self.read_pdf_file(filepath), 'pdf'
        
        # Word documents
        elif ext in ['.docx', '.doc']:
            return self.read_docx_file(filepath), 'docx'
        
        # Image files (OCR)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return self.read_image_file(filepath), 'image'
        
        else:
            return f"Error: Unsupported file type - {ext}", "unknown"
    
    def extract_student_id_from_filename(self, filename: str) -> str:
        """
        Extract student ID from filename
        Supports formats like: student_001.txt, 001_assignment.pdf, John_Doe_12345.docx
        """
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Try to find numeric ID
        import re
        numbers = re.findall(r'\d+', name)
        if numbers:
            return f"student_{numbers[0]}"
        
        # Otherwise use the filename itself
        return name.replace(' ', '_').replace('-', '_')
    
    def load_from_file(self, filepath: str, student_id: Optional[str] = None):
        """
        Load a single assignment file
        
        Args:
            filepath: Path to the assignment file
            student_id: Optional student ID (auto-extracted if not provided)
        """
        content, filetype = self.read_file(filepath)
        
        if content.startswith("Error:"):
            print(f"Failed to load {filepath}: {content}")
            return None
        
        # Extract student ID from filename if not provided
        if not student_id:
            filename = os.path.basename(filepath)
            student_id = self.extract_student_id_from_filename(filename)
        
        return self.add_submission(
            student_id=student_id,
            content=content,
            metadata={
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'filetype': filetype
            }
        )
    
    def load_from_directory(self, directory: str, pattern: str = "*.*"):
        """
        Load all matching files from a directory
        
        Args:
            directory: Path to directory containing assignments
            pattern: File pattern to match (e.g., "*.pdf", "*.txt", "assignment_*.docx")
        
        Returns:
            Number of files loaded
        """
        if not os.path.exists(directory):
            print(f"Error: Directory not found - {directory}")
            return 0
        
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        
        if not files:
            print(f"No files found matching pattern: {search_path}")
            return 0
        
        print(f"Found {len(files)} file(s) in {directory}")
        loaded = 0
        
        for filepath in files:
            if os.path.isfile(filepath):
                result = self.load_from_file(filepath)
                if result:
                    loaded += 1
                    print(f"✓ Loaded: {os.path.basename(filepath)}")
        
        print(f"\nSuccessfully loaded {loaded}/{len(files)} files")
        return loaded
    
    def add_submission(self, student_id: str, content: str, metadata: Dict = None):
        """Add a student submission to DataFrame"""
        submission = {
            'student_id': student_id,
            'content': content,
            'timestamp': datetime.now(),
            'hash': hashlib.md5(content.encode()).hexdigest(),
            'word_count': len(content.split()),
            'char_count': len(content)
        }
        
        # Add metadata columns if provided
        if metadata:
            submission.update(metadata)
        
        # Append to DataFrame
        self.submissions_df = pd.concat([
            self.submissions_df, 
            pd.DataFrame([submission])
        ], ignore_index=True)
        
        return submission
    
    def check_plagiarism(self) -> pd.DataFrame:
        """
        Check for plagiarism using NumPy for efficient matrix operations
        
        Returns:
            DataFrame with flagged pairs
        """
        if len(self.submissions_df) < 2:
            return pd.DataFrame()
        
        # Extract content
        contents = self.submissions_df['content'].tolist()
        
        # Calculate TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(contents)
        
        # Calculate cosine similarity using NumPy
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Use NumPy to find similar pairs efficiently
        n = len(self.submissions_df)
        flagged_data = []
        
        # Create upper triangular mask (avoid duplicate pairs)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Find indices where similarity exceeds threshold
        flagged_indices = np.argwhere(
            (similarity_matrix >= self.similarity_threshold) & mask
        )
        
        for i, j in flagged_indices:
            flagged_data.append({
                'student_1': self.submissions_df.iloc[i]['student_id'],
                'student_2': self.submissions_df.iloc[j]['student_id'],
                'similarity_score': similarity_matrix[i, j],
                'timestamp': datetime.now()
            })
        
        self.plagiarism_df = pd.DataFrame(flagged_data)
        return self.plagiarism_df
    
    def grade_assignment(self, content: str, rubric: Dict, student_id: str = None) -> Dict:
        """Grade assignment and store in DataFrame"""
        content_lower = content.lower()
        results = {
            'total_score': 0.0,
            'max_score': 0.0,
            'criteria_scores': {},
            'feedback': []
        }
        
        for criterion, details in rubric.items():
            max_points = float(details.get('points', 0))
            keywords = details.get('keywords', [])
            required = details.get('required', False)
            
            results['max_score'] += max_points
            
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            match_ratio = matches / len(keywords) if keywords else 0
            
            # Calculate score
            if required and matches == 0:
                score = 0.0
                feedback = f"Missing required criterion: {criterion}"
            else:
                score = match_ratio * max_points
                feedback = f"{criterion}: {matches}/{len(keywords)} keywords found"
            
            results['total_score'] += score
            results['criteria_scores'][criterion] = score
            results['feedback'].append(feedback)
        
        results['percentage'] = (results['total_score'] / results['max_score'] * 100) if results['max_score'] > 0 else 0
        results['letter_grade'] = self._calculate_letter_grade(results['percentage'])
        
        # Store in grades DataFrame if student_id provided
        if student_id:
            grade_entry = {
                'student_id': student_id,
                'total_score': results['total_score'],
                'max_score': results['max_score'],
                'percentage': results['percentage'],
                'letter_grade': results['letter_grade'],
                'timestamp': datetime.now()
            }
            self.grades_df = pd.concat([
                self.grades_df,
                pd.DataFrame([grade_entry])
            ], ignore_index=True)
        
        return results
    
    def grade_all_submissions(self, rubric: Dict):
        """Grade all submissions at once"""
        for idx, row in self.submissions_df.iterrows():
            self.grade_assignment(
                row['content'], 
                rubric, 
                student_id=row['student_id']
            )
    
    def get_statistics(self) -> Dict:
        """Calculate comprehensive statistics using Pandas & NumPy"""
        if self.grades_df.empty:
            return {"error": "No grades available"}
        
        stats = {
            # Basic stats using Pandas
            'total_submissions': len(self.submissions_df),
            'graded_submissions': len(self.grades_df),
            'plagiarism_cases': len(self.plagiarism_df),
            
            # Grade statistics using NumPy
            'mean_score': float(np.mean(self.grades_df['percentage'])),
            'median_score': float(np.median(self.grades_df['percentage'])),
            'std_dev': float(np.std(self.grades_df['percentage'])),
            'min_score': float(np.min(self.grades_df['percentage'])),
            'max_score': float(np.max(self.grades_df['percentage'])),
            
            # Percentiles
            'percentile_25': float(np.percentile(self.grades_df['percentage'], 25)),
            'percentile_75': float(np.percentile(self.grades_df['percentage'], 75)),
            
            # Grade distribution using Pandas value_counts
            'grade_distribution': self.grades_df['letter_grade'].value_counts().to_dict(),
            
            # Word count statistics
            'avg_word_count': float(self.submissions_df['word_count'].mean()),
            'word_count_std': float(self.submissions_df['word_count'].std())
        }
        
        return stats
    
    def get_top_performers(self, n: int = 5) -> pd.DataFrame:
        """Get top N performers using Pandas sorting"""
        return self.grades_df.nlargest(n, 'percentage')[
            ['student_id', 'percentage', 'letter_grade']
        ]
    
    def get_low_performers(self, threshold: float = 60) -> pd.DataFrame:
        """Get students below threshold"""
        return self.grades_df[self.grades_df['percentage'] < threshold][
            ['student_id', 'percentage', 'letter_grade']
        ]
    
    def get_outliers(self) -> pd.DataFrame:
        """Detect outlier scores using statistical methods"""
        if len(self.grades_df) < 3:
            return pd.DataFrame()
        
        # Use z-score method with NumPy
        scores = self.grades_df['percentage'].values
        z_scores = np.abs((scores - np.mean(scores)) / np.std(scores))
        
        # Outliers are typically z-score > 2
        outlier_mask = z_scores > 2
        
        outliers = self.grades_df[outlier_mask].copy()
        outliers['z_score'] = z_scores[outlier_mask]
        
        return outliers[['student_id', 'percentage', 'z_score']]
    
    def _calculate_letter_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade"""
        if percentage >= 90:
            return 'A'
        elif percentage >= 80:
            return 'B'
        elif percentage >= 70:
            return 'C'
        elif percentage >= 60:
            return 'D'
        else:
            return 'F'
    
    def export_to_csv(self, filename: str = "assignment_results.csv"):
        """Export results to CSV using Pandas"""
        try:
            # Merge submissions with grades
            merged = self.submissions_df.merge(
                self.grades_df, 
                on='student_id', 
                how='left'
            )
            
            # Select relevant columns
            export_df = merged[[
                'student_id', 'filename', 'filetype', 'word_count', 
                'total_score', 'max_score', 'percentage', 'letter_grade'
            ]]
            
            export_df.to_csv(filename, index=False)
            return f"Exported to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"
    
    def export_to_excel(self, filename: str = "assignment_results.xlsx"):
        """Export comprehensive report to Excel with multiple sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Grades
                merged = self.submissions_df.merge(
                    self.grades_df, 
                    on='student_id', 
                    how='left'
                )
                merged[['student_id', 'filename', 'word_count', 'percentage', 'letter_grade']].to_excel(
                    writer, sheet_name='Grades', index=False
                )
                
                # Sheet 2: Plagiarism
                if not self.plagiarism_df.empty:
                    self.plagiarism_df.to_excel(
                        writer, sheet_name='Plagiarism', index=False
                    )
                
                # Sheet 3: Statistics
                stats_df = pd.DataFrame([self.get_statistics()]).T
                stats_df.columns = ['Value']
                stats_df.to_excel(writer, sheet_name='Statistics')
            
            return f"Exported to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"
    
    def send_to_teams(self, webhook_url: str, include_stats: bool = True) -> bool:
        """Send comprehensive report to Microsoft Teams"""
        try:
            stats = self.get_statistics() if include_stats else {}
            
            # Format message for Teams
            teams_message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Assignment Checker Results",
                "themeColor": "0078D4",
                "title": "Assignment Checker Report",
                "sections": [{
                    "activityTitle": "Grading Complete",
                    "facts": [
                        {"name": "Total Submissions", "value": str(stats.get('total_submissions', 0))},
                        {"name": "Graded", "value": str(stats.get('graded_submissions', 0))},
                        {"name": "Plagiarism Cases", "value": str(stats.get('plagiarism_cases', 0))},
                        {"name": "Average Score", "value": f"{stats.get('mean_score', 0):.1f}%"},
                        {"name": "Median Score", "value": f"{stats.get('median_score', 0):.1f}%"},
                        {"name": "Std Deviation", "value": f"{stats.get('std_dev', 0):.1f}"},
                    ],
                    "text": f"Grade Distribution: {stats.get('grade_distribution', {})}"
                }]
            }
            
            response = requests.post(webhook_url, json=teams_message)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending to Teams: {str(e)}")
            return False
    
    def generate_report(self) -> str:
        """Generate comprehensive text report with statistics"""
        report = []
        report.append("ASSIGNMENT CHECKER REPORT")
        report.append("")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Statistics section
        stats = self.get_statistics()
        report.append("STATISTICS")
        report.append(f"Total Submissions: {stats.get('total_submissions', 0)}")
        report.append(f"Graded: {stats.get('graded_submissions', 0)}")
        report.append(f"Average Score: {stats.get('mean_score', 0):.2f}%")
        report.append(f"Median Score: {stats.get('median_score', 0):.2f}%")
        report.append(f"Standard Deviation: {stats.get('std_dev', 0):.2f}")
        report.append(f"Range: {stats.get('min_score', 0):.1f}% - {stats.get('max_score', 0):.1f}%")
        report.append("")
        
        # Grade distribution
        report.append("GRADE DISTRIBUTION")
        for grade, count in stats.get('grade_distribution', {}).items():
            report.append(f"  {grade}: {count} students")
        report.append("")
        
        # Plagiarism section
        report.append("PLAGIARISM DETECTION")
        if not self.plagiarism_df.empty:
            report.append(f"WARNING: {len(self.plagiarism_df)} similar submission(s) found:")
            for _, row in self.plagiarism_df.iterrows():
                report.append(f"  {row['student_1']} and {row['student_2']}")
                report.append(f"    Similarity: {row['similarity_score']:.2%}")
        else:
            report.append("No suspicious similarities detected")
        report.append("")
        
        # Top performers
        report.append("TOP PERFORMERS")
        top = self.get_top_performers(3)
        for _, row in top.iterrows():
            report.append(f"  {row['student_id']}: {row['percentage']:.1f}% ({row['letter_grade']})")
        report.append("")
        
        # Low performers
        low = self.get_low_performers(60)
        if not low.empty:
            report.append("NEEDS ATTENTION (Below 60%)")
            for _, row in low.iterrows():
                report.append(f"  {row['student_id']}: {row['percentage']:.1f}% ({row['letter_grade']})")
            report.append("")
        
        return "\n".join(report)


# Main execution
if __name__ == "__main__":
    # Initialize checker
    print("=" * 70)
    print("AI ASSIGNMENT CHECKER")
    print("=" * 70)
    
    threshold = input("\nSimilarity threshold for plagiarism (0-1) [default 0.75]: ").strip()
    threshold = float(threshold) if threshold else 0.75
    
    checker = AssignmentChecker(similarity_threshold=threshold)
    
    # Get rubric from user
    print("\n" + "=" * 70)
    print("RUBRIC CONFIGURATION")
    print("=" * 70)
    
    use_default = input("\nUse default rubric? (y/n): ").strip().lower()
    
    if use_default == 'y':
        rubric = {
            "Introduction": {
                "points": 20,
                "keywords": ["introduction", "thesis", "overview", "purpose"],
                "required": True
            },
            "Analysis": {
                "points": 40,
                "keywords": ["analyze", "evidence", "data", "research", "study"],
                "required": True
            },
            "Conclusion": {
                "points": 20,
                "keywords": ["conclusion", "summary", "findings", "results"],
                "required": True
            },
            "Citations": {
                "points": 20,
                "keywords": ["reference", "cited", "source", "bibliography"],
                "required": False
            }
        }
        print("✓ Using default rubric")
    else:
        rubric_path = input("Enter path to rubric JSON file: ").strip()
        try:
            with open(rubric_path, 'r') as f:
                rubric = json.load(f)
            print("✓ Rubric loaded successfully")
        except Exception as e:
            print(f"✗ Error loading rubric: {e}")
            print("Using default rubric instead")
            rubric = {
                "Content": {"points": 50, "keywords": [], "required": False},
                "Quality": {"points": 50, "keywords": [], "required": False}
            }
    
    # Get directory path from user
    print("\n" + "=" * 70)
    print("FILE LOADING")
    print("=" * 70)
    
    directory = input("\nEnter path to assignments folder: ").strip()
    
    if not directory:
        print("\n⚠️  No directory provided. Using current directory.")
        directory = "."
    
    if not os.path.exists(directory):
        print(f"✗ Directory not found: {directory}")
        exit()
    
    # Optional: Ask for specific file pattern
    use_pattern = input("Filter by pattern? (e.g., *.pdf, assignment_*.txt) [Enter for all]: ").strip()
    pattern = use_pattern if use_pattern else "*.*"
    
    # Load all files from directory
    print(f"\nSearching for files in: {os.path.abspath(directory)}")
    print(f"Pattern: {pattern}\n")
    
    loaded_count = checker.load_from_directory(directory, pattern)
    
    if loaded_count == 0:
        print("\n❌ No files loaded. Exiting.")
        exit()
    
    # Check plagiarism
    print("\n" + "=" * 70)
    print("PLAGIARISM CHECK")
    print("=" * 70)
    print("Analyzing submissions...")
    
    flagged = checker.check_plagiarism()
    print(f"✓ Found {len(flagged)} similar pair(s)")
    
    # Grade all assignments
    print("\n" + "=" * 70)
    print("GRADING ASSIGNMENTS")
    print("=" * 70)
    print("Applying rubric...")
    
    checker.grade_all_submissions(rubric)
    print(f"✓ Graded {len(checker.grades_df)} submission(s)")
    
    # Display comprehensive report
    print("\n" + checker.generate_report())
    
    # Export results
    print("\n" + "=" * 70)
    print("EXPORT OPTIONS")
    print("=" * 70)
    
    export_csv = input("\nExport to CSV? (y/n): ").strip().lower()
    if export_csv == 'y':
        csv_name = input("CSV filename [default: assignment_results.csv]: ").strip()
        csv_name = csv_name if csv_name else "assignment_results.csv"
        print(checker.export_to_csv(csv_name))
    
    export_excel = input("Export to Excel? (y/n): ").strip().lower()
    if export_excel == 'y':
        excel_name = input("Excel filename [default: assignment_results.xlsx]: ").strip()
        excel_name = excel_name if excel_name else "assignment_results.xlsx"
        print(checker.export_to_excel(excel_name))
    
    # Optional: Send to Teams
    send_teams = input("\nSend report to Microsoft Teams? (y/n): ").strip().lower()
    if send_teams == 'y':
        webhook = input("Enter Teams webhook URL: ").strip()
        if webhook:
            print("Sending to Teams...")
            if checker.send_to_teams(webhook):
                print("✓ Successfully sent to Teams!")
            else:
                print("✗ Failed to send to Teams")
        else:
            print("✗ No webhook URL provided")
    
    print("\n" + "=" * 70)
    print("✓ All done!")
    print("=" * 70)
