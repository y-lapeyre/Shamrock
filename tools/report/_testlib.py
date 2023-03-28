
class TestInstance:
    def __init__(self, result):
        self.type = result["type"]
        self.name = result["name"]
        self.compute_queue = result["compute_queue"]
        self.alt_queue = result["alt_queue"]
        self.world_rank = result["world_rank"]
        self.asserts = result["asserts"]
        self.test_data = result["test_data"]

    def get_test_dataset(self,dataset_name, table_name):
        for d in self.test_data:
            if(d["dataset_name"] == dataset_name):
                for t in d["dataset"]:
                    if(t["name"] == table_name):
                        return t["data"]

        return None

    

class TestResults:

    def __init__(self, index, result):
        self.commit_hash = result["commit_hash"]
        self.world_size = result["world_size"]
        self.compiler = result["compiler"]
        self.comp_args = result["comp_args"]
        self.index = index
        self.results = result["results"]

    def get_config_str(self) -> str:
        buf = ""
        buf += r"\subsection{"
        buf += f"Config {self.index}"
        buf += r"}" + "\n\n" + r"\begin{itemize}"
        
        buf += f"""
        \\item Commit Hash : {self.commit_hash}
        \\item World size : {self.world_size}
        \\item Compiler : {self.compiler}
        """

        buf += r"\end{itemize}"

        return buf

    
    def get_test_instances(self, type, name):
        instances = []
        for r in self.results:
            if r["type"] == type and r["name"] == name:
                instances.append(TestInstance(r))

        return instances


