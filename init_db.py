import pylidc as pl
import os
import shutil
from pathlib import Path
import sqlite3
import glob
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pydicom
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Get the absolute path to the LIDC-IDRI directory
data_dir = os.path.abspath('D:/BasesDatos/LIDC-IDRI-Preprocessing/LIDC-IDRI')

# Get the path to the pylidc.sqlite file
pylidc_db_path = Path(pl.__file__).parent / 'pylidc.sqlite'

# Create a backup of the original database
if pylidc_db_path.exists():
    backup_path = pylidc_db_path.with_suffix('.sqlite.bak')
    shutil.copy2(pylidc_db_path, backup_path)
    print(f"Created backup of original database at {backup_path}")

# Verify that the LIDC-IDRI directory exists and contains patient data
if not os.path.exists(data_dir):
    raise RuntimeError(f"LIDC-IDRI directory not found at {data_dir}")

patient_dirs = glob.glob(os.path.join(data_dir, "LIDC-IDRI-*"))
if not patient_dirs:
    raise RuntimeError(f"No patient directories found in {data_dir}")

print(f"Found {len(patient_dirs)} patient directories")

# Initialize the database
print("Initializing database...")
engine = create_engine('sqlite:///' + str(pylidc_db_path))
Session = sessionmaker(bind=engine)
session = Session()

# Create tables if they don't exist
from pylidc._Base import Base
Base.metadata.create_all(engine)

# Import patient data and annotations
print("Importing patient data and annotations...")
for patient_dir in tqdm(patient_dirs):
    patient_id = os.path.basename(patient_dir)
    
    # Find DICOM files and XML files
    dicom_files = []
    xml_files = []
    for root, _, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
            elif file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    if not dicom_files:
        continue
    
    try:
        # Read first DICOM file to get patient info
        dcm = pydicom.dcmread(dicom_files[0])
        
        # Check if scan already exists
        existing_scan = session.query(pl.Scan).filter_by(patient_id=patient_id).first()
        if existing_scan:
            print(f"Scan for {patient_id} already exists, skipping...")
            continue
            
        # Create Scan record using the correct method
        scan = pl.Scan()
        scan.study_instance_uid = str(dcm.StudyInstanceUID)
        scan.series_instance_uid = str(dcm.SeriesInstanceUID)
        scan.slice_thickness = float(dcm.SliceThickness)
        scan.pixel_spacing = float(dcm.PixelSpacing[0])
        scan.contrast_used = bool(dcm.get('ContrastBolusAgent', False))
        scan.is_from_initial = True
        
        # Set the patient_id using the internal method
        scan._patient_id = patient_id
        
        session.add(scan)
        session.commit()
        
        # Process XML files for annotations
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Process each nodule annotation
                for reading in root.findall('.//readingSession'):
                    for nodule in reading.findall('.//unblindedReadNodule'):
                        # Create Annotation record
                        annotation = pl.Annotation(
                            scan_id=scan.id,
                            subtlety=int(nodule.find('subtlety').text),
                            internalStructure=int(nodule.find('internalStructure').text),
                            calcification=int(nodule.find('calcification').text),
                            sphericity=int(nodule.find('sphericity').text),
                            margin=int(nodule.find('margin').text),
                            lobulation=int(nodule.find('lobulation').text),
                            spiculation=int(nodule.find('spiculation').text),
                            texture=int(nodule.find('texture').text),
                            malignancy=int(nodule.find('malignancy').text)
                        )
                        session.add(annotation)
                        
                        # Process contours
                        for contour in nodule.findall('.//roi'):
                            inclusion = contour.find('inclusion').text == 'TRUE'
                            z_pos = float(contour.find('imageZposition').text)
                            
                            # Find corresponding DICOM file
                            dicom_file = None
                            for dcm_file in dicom_files:
                                dcm = pydicom.dcmread(dcm_file)
                                if abs(float(dcm.ImagePositionPatient[2]) - z_pos) < 0.001:
                                    dicom_file = os.path.basename(dcm_file)
                                    break
                            
                            if dicom_file:
                                # Create Contour record
                                contour_obj = pl.Contour(
                                    annotation_id=annotation.id,
                                    inclusion=inclusion,
                                    image_z_position=z_pos,
                                    dicom_file_name=dicom_file,
                                    coords=contour.find('edgeMap').text
                                )
                                session.add(contour_obj)
                
                session.commit()
                
            except Exception as e:
                print(f"Error processing XML file {xml_file}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
        continue

print("Database initialized and populated successfully!") 