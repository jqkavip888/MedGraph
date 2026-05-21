# 构建一个neo4j类
import os
import json
from neo4j import GraphDatabase
from neo4j_config import NEO4J_CONFIG

driver = GraphDatabase.driver(**NEO4J_CONFIG)


# 定义工作路径
class MedicalGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/medical.json')
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)


    # 清空旧数据
    def delete_all(self):
        with self.driver.session() as session:
            print("正在清空旧数据...")
            session.run("MATCH (n) DETACH DELETE n")

    # 构建schema，定义节点，与节点的关系
    def read_node(self):
        drug = set()
        food = set()
        disease = set()
        symptom = set()

        # 构建节点之间的关系，暂时使用空list
        rels_recommandeat = []  # 疾病-食物推荐
        rels_recommanddrug = []  # 疾病-用药推荐
        rels_symptom = []  # 疾病-症状

        # 读取数据集
        count = 0
        for data in open(self.data_path, encoding='utf-8'):
            count += 1
            disease_dict = {}
            if count % 500 == 0:
                print('count = :', count)

            data_json = json.loads(data)
            current_disease = data_json['name']
            disease.add(current_disease)

            # 1. 判定症状在数据集中，则添加关系
            if 'symptom' in data_json:
                current_symptoms = data_json['symptom']  # 获取当前疾病的症状
                for s in current_symptoms:
                    symptom.add(s)  # 加入全局集合用于建节点
                    rels_symptom.append([current_disease, s])  # 只建立当前疾病的关系

            # 2. 处理药品关系
            if 'recommand_drug' in data_json:
                current_drugs = data_json['recommand_drug']
                for d in current_drugs:
                    drug.add(d)
                    rels_recommanddrug.append([current_disease, d])

            # 3. 处理食物关系
            if 'recommand_eat' in data_json:
                current_foods = data_json['recommand_eat']
                for f in current_foods:
                    food.add(f)
                    rels_recommandeat.append([current_disease, f])

        return drug, food, symptom, disease, rels_recommandeat, rels_recommanddrug, rels_symptom

    # 在neo4j中创建知识图谱节点和节点关系
    def create_graphnodes_and_graphrels(self):
        # 节点封装
        Drug, Food, Symptom, Disease, rels_recommandeat, rels_recommanddrug, rels_symptom = self.read_node()

        print('Drugs:', len(Drug))
        print('Foods:', len(Food))
        print('Symptoms:', len(Symptom))
        print('Diseases', len(Disease))
        print('Rels_recommandeat:', len(rels_recommandeat))
        print('Rels_recommanddrug:', len(rels_recommanddrug))
        print('Rels_symptoms:', len(rels_symptom))

        driver = GraphDatabase.driver(**NEO4J_CONFIG)

        with driver.session() as session:
            # 创建疾病节点
            print('开始创建疾病节点...')
            for d in Disease:
                cypher = 'MERGE (a:Disease{name:%r}) RETURN a' % d
                session.run(cypher)

            print('开始创建药品drug节点...')
            for d in Drug:
                cypher = 'MERGE (a:Drug{name:%r}) RETURN a' % d
                session.run(cypher)

            print('开始创建食品food节点...')
            for f in Food:
                cypher = 'MERGE (a:Food{name:%r}) RETURN a' % f
                session.run(cypher)

            print('开始创建症状symptom节点...')
            for s in Symptom:
                cypher = 'MERGE (a:Symptom{name:%r}) RETURN a' % s
                session.run(cypher)

        # 创建实体关系边的规则
        self.create_relationship('Disease', 'Food', rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationship('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '推荐药品')
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')

    # 开始创建实体关系边
    def create_relationship(self, start_node, end_node, edges, rels_type, rels_name):
        # 去重处理
        set_edge = []
        for edge in edges:
            set_edge.append('###'.join(edge))
        num_edges = len(set_edge)
        print('num_edges:', num_edges)

        driver = GraphDatabase.driver(**NEO4J_CONFIG)
        with driver.session() as session:
            for edge in set(set_edge):
                edge = edge.split('###')
                # 取出两个值作为两个节点
                p = edge[0]
                q = edge[1]
                cypher = "MATCH (p:%s),(q:%s) WHERE p.name='%s' AND q.name='%s' MERGE (p)-[rel:%s{name:'%s'}]->(q)" \
                         % (start_node, end_node, p, q, rels_type, rels_name)


                try:
                    session.run(cypher)
                except Exception as e:
                    print(f"Error creating relation between {p} and {q}: {e}")
        return



if __name__ == '__main__':
    mg = MedicalGraph()
    print('创建知识图谱各个节点与节点关系...')
    mg.delete_all()
    mg.create_graphnodes_and_graphrels()
