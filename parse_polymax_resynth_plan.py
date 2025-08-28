#!/usr/bin/env python3
"""
PolyMAX Resynthesis Plan Parser

Utility to parse and validate the polymax_resynth_plan.xml configuration file.
This demonstrates how the XML configuration integrates with the PolyMAX workflow.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from pathlib import Path


class PolyMAXResynthPlanParser:
    """Parser for PolyMAX resynthesis plan XML configuration."""
    
    def __init__(self, xml_path: str = "polymax_resynth_plan.xml"):
        """Initialize parser with XML file path."""
        self.xml_path = Path(xml_path)
        self.tree = None
        self.root = None
        self.load_xml()
    
    def load_xml(self) -> None:
        """Load and parse the XML file."""
        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()
            if self.root.tag != "PolyMAXResynthPlan":
                raise ValueError(f"Invalid root element: {self.root.tag}")
        except ET.ParseError as e:
            raise ValueError(f"XML parsing error: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")
    
    def get_metadata(self) -> Dict[str, str]:
        """Extract metadata from the XML configuration."""
        metadata = {}
        metadata_elem = self.root.find('metadata')
        if metadata_elem is not None:
            for child in metadata_elem:
                metadata[child.tag] = child.text
        return metadata
    
    def get_parameter_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract parameter groups and their configurations."""
        groups = {}
        param_mapping = self.root.find('parameter_mapping/synthesis_parameters')
        
        if param_mapping is not None:
            for group_elem in param_mapping.findall('parameter_group'):
                group_name = group_elem.get('name')
                group_priority = group_elem.get('priority')
                
                parameters = []
                for param_elem in group_elem.findall('parameter'):
                    param_info = {
                        'name': param_elem.get('name'),
                        'index': int(param_elem.get('index')),
                        'type': param_elem.get('type'),
                        'range': param_elem.get('range'),
                        'weight': float(param_elem.get('weight')),
                        'importance': param_elem.get('resynthesis_importance')
                    }
                    parameters.append(param_info)
                
                groups[group_name] = {
                    'priority': group_priority,
                    'parameters': parameters
                }
        
        return groups
    
    def get_workflow_stages(self) -> List[Dict[str, Any]]:
        """Extract resynthesis workflow stages."""
        stages = []
        workflow = self.root.find('resynthesis_workflow/stages')
        
        if workflow is not None:
            for stage_elem in workflow.findall('stage'):
                stage_info = {
                    'name': stage_elem.get('name'),
                    'order': int(stage_elem.get('order')),
                    'description': stage_elem.find('description').text if stage_elem.find('description') is not None else "",
                    'operations': []
                }
                
                operations_elem = stage_elem.find('operations')
                if operations_elem is not None:
                    for op_elem in operations_elem.findall('operation'):
                        op_info = {
                            'name': op_elem.get('name'),
                            'timeout': op_elem.get('timeout'),
                            'attributes': dict(op_elem.attrib)
                        }
                        stage_info['operations'].append(op_info)
                
                stages.append(stage_info)
        
        return sorted(stages, key=lambda x: x['order'])
    
    def get_model_configuration(self) -> Dict[str, Any]:
        """Extract model configuration settings."""
        config = {}
        model_config = self.root.find('model_configuration')
        
        if model_config is not None:
            for section in model_config:
                section_config = {}
                for child in section:
                    try:
                        # Try to convert to appropriate type
                        value = child.text
                        if value.lower() in ['true', 'false']:
                            section_config[child.tag] = value.lower() == 'true'
                        elif '.' in value:
                            section_config[child.tag] = float(value)
                        elif value.isdigit():
                            section_config[child.tag] = int(value)
                        else:
                            section_config[child.tag] = value
                    except (ValueError, AttributeError):
                        section_config[child.tag] = child.text
                
                config[section.tag] = section_config
        
        return config
    
    def get_critical_parameters(self) -> List[Dict[str, Any]]:
        """Get parameters marked as critical for resynthesis."""
        critical_params = []
        groups = self.get_parameter_groups()
        
        for group_name, group_info in groups.items():
            for param in group_info['parameters']:
                if param['importance'] == 'critical':
                    param['group'] = group_name
                    critical_params.append(param)
        
        return sorted(critical_params, key=lambda x: x['weight'], reverse=True)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the XML configuration and return validation results."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check metadata
            metadata = self.get_metadata()
            required_metadata = ['version', 'plugin_name', 'plugin_id']
            for req in required_metadata:
                if req not in metadata:
                    validation['errors'].append(f"Missing required metadata: {req}")
                    validation['valid'] = False
            
            # Check parameter groups
            groups = self.get_parameter_groups()
            if not groups:
                validation['errors'].append("No parameter groups found")
                validation['valid'] = False
            
            total_params = sum(len(group['parameters']) for group in groups.values())
            critical_params = len(self.get_critical_parameters())
            
            validation['statistics'] = {
                'total_parameter_groups': len(groups),
                'total_parameters': total_params,
                'critical_parameters': critical_params,
                'critical_parameter_ratio': critical_params / total_params if total_params > 0 else 0
            }
            
            # Check workflow stages
            stages = self.get_workflow_stages()
            if len(stages) < 3:
                validation['warnings'].append("Less than 3 workflow stages defined")
            
            validation['statistics']['workflow_stages'] = len(stages)
            
            # Check model configuration
            model_config = self.get_model_configuration()
            if not model_config:
                validation['warnings'].append("No model configuration found")
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            validation['valid'] = False
        
        return validation
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of the configuration."""
        print("=" * 60)
        print("POLYMAX RESYNTHESIS PLAN SUMMARY")
        print("=" * 60)
        
        # Metadata
        metadata = self.get_metadata()
        print(f"\nPlugin: {metadata.get('plugin_name', 'Unknown')}")
        print(f"Version: {metadata.get('version', 'Unknown')}")
        print(f"Description: {metadata.get('description', 'No description')}")
        
        # Parameter groups
        groups = self.get_parameter_groups()
        print(f"\nParameter Groups ({len(groups)}):")
        for group_name, group_info in groups.items():
            param_count = len(group_info['parameters'])
            priority = group_info['priority']
            print(f"  • {group_name} ({priority}): {param_count} parameters")
        
        # Critical parameters
        critical_params = self.get_critical_parameters()
        print(f"\nCritical Parameters ({len(critical_params)}):")
        for param in critical_params[:5]:  # Show top 5
            print(f"  • {param['name']} (weight: {param['weight']}, group: {param['group']})")
        if len(critical_params) > 5:
            print(f"  ... and {len(critical_params) - 5} more")
        
        # Workflow stages
        stages = self.get_workflow_stages()
        print(f"\nWorkflow Stages ({len(stages)}):")
        for stage in stages:
            op_count = len(stage['operations'])
            print(f"  {stage['order']}. {stage['name']}: {op_count} operations")
        
        # Model configuration
        model_config = self.get_model_configuration()
        if 'flow_model' in model_config:
            flow_config = model_config['flow_model']
            print(f"\nFlow Model Configuration:")
            print(f"  • Architecture: {flow_config.get('architecture', 'Unknown')}")
            print(f"  • Flow Type: {flow_config.get('flow_type', 'Unknown')}")
            print(f"  • Flow Length: {flow_config.get('flow_length', 'Unknown')}")
            print(f"  • Latent Dims: {flow_config.get('latent_dims', 'Unknown')}")
        
        # Validation
        validation = self.validate_configuration()
        print(f"\nValidation Status: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  • {error}")
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        stats = validation.get('statistics', {})
        if stats:
            print(f"\nStatistics:")
            print(f"  • Total Parameters: {stats.get('total_parameters', 0)}")
            print(f"  • Critical Parameters: {stats.get('critical_parameters', 0)} ({stats.get('critical_parameter_ratio', 0):.1%})")
            print(f"  • Workflow Stages: {stats.get('workflow_stages', 0)}")


def main():
    """Main function for command-line usage."""
    try:
        parser = PolyMAXResynthPlanParser()
        parser.print_summary()
        
        print("\n" + "=" * 60)
        print("CONFIGURATION PARSING SUCCESSFUL")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())